#pragma once
#ifndef I2SMIC_HPP
#define I2SMIC_HPP

#include "board/board_config.h"
#include "hal/sensor.hpp"

#include <driver/gpio.h>
#include <driver/i2s_pdm.h>
#include <driver/i2s_std.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

namespace porting {

using namespace hal;

class SensorI2SMic final: public Sensor
{
public:
    static inline core::ConfigObjectMap DEFAULT_CONFIGS() noexcept
    {
#if BOARD_USE_PDM_MODE
        return { CONFIG_OBJECT_DECL_INTEGER("clk_pin", "Clock pin number", BOARD_PDM_CLK_PIN, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("din_pin", "Data pin number", BOARD_PDM_DIN_PIN, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("sr", "PCM sample rate in Hz", BOARD_RAW_SAMPLE_RATE, 8000, 48000),
            CONFIG_OBJECT_DECL_INTEGER("channels", "Number of channels", 1, 1, 1),
            CONFIG_OBJECT_DECL_INTEGER("buffered_duration", "Time duration of buffered data for DMA in seconds", 2, 1,
                5) };
#else
        return { CONFIG_OBJECT_DECL_INTEGER("bclk_pin", "I2S BCLK pin", BOARD_I2S_BCLK_PIN, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("ws_pin", "I2S WS/LRCLK pin", BOARD_I2S_WS_PIN, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("sr", "PCM sample rate in Hz", BOARD_RAW_SAMPLE_RATE, 8000, 48000),
            CONFIG_OBJECT_DECL_INTEGER("channels", "Number of channels", 1, 1, 1),
            CONFIG_OBJECT_DECL_INTEGER("buffered_duration", "Time duration of buffered data for DMA in seconds", 2, 1,
                5) };
#endif
    }

    SensorI2SMic() noexcept : Sensor(Info(2, "I2S Microphone", Type::Microphone, { DEFAULT_CONFIGS() })) { }

    core::Status init() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status != Status::Uninitialized)
        {
            return STATUS(ENXIO, "Sensor is already initialized or in an invalid state");
        }

        const size_t sr = _info.configs["sr"].getValue<int>();
        _channels = _info.configs["channels"].getValue<int>();
        _buffered_duration = _info.configs["buffered_duration"].getValue<int>();
        _data_buffer_capacity_frames = _buffered_duration * sr;

        if (!_data_buffer) [[likely]]
        {
            _data_buffer_capacity_bytes = _data_buffer_capacity_frames * _channels * sizeof(int16_t);
            _data_buffer = std::shared_ptr<std::byte[]>(new std::byte[_data_buffer_capacity_bytes]);
            if (!_data_buffer)
            {
                LOG(ERROR, "Failed to allocate data buffer, size: %zu bytes", _data_buffer_capacity_bytes);
                return STATUS(ENOMEM, "Failed to allocate data buffer");
            }
            LOG(DEBUG, "Allocated data buffer of size: %zu bytes at %p", _data_buffer_capacity_bytes,
                _data_buffer.get());
        }

        if (!_rx_chan) [[likely]]
        {
            const uint32_t frames_count = BOARD_DMA_FRAME_COUNT;
            const uint32_t slots_count = _data_buffer_capacity_frames / frames_count;
            LOG(DEBUG,
                "Creating I2S channel with sample rate %d, buffered frames %d, sync frame count %ld, slots count %ld",
                sr, _data_buffer_capacity_frames, frames_count, slots_count);

#if BOARD_USE_PDM_MODE
            _rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
            LOG(INFO, "Configuring I2S: PDM mode, Master role");
#else
            _rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, BOARD_I2S_ROLE);
            LOG(INFO, "Configuring I2S: STD mode, Role=%d", BOARD_I2S_ROLE);
#endif

            _rx_chan_cfg.dma_desc_num = slots_count;
            _rx_chan_cfg.dma_frame_num = frames_count;

            auto ret = i2s_new_channel(&_rx_chan_cfg, NULL, &_rx_chan);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to create I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to create I2S channel");
            }

            ret = i2s_channel_register_event_callback(_rx_chan, &_event_cbs, this);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to register I2S channel event callback: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to register I2S channel event callback");
            }
        }

#if BOARD_USE_PDM_MODE
        {
            _pdm_rx_cfg.clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(sr);
            _pdm_rx_cfg.slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                (_info.configs["channels"].getValue<int>() == 1 ? I2S_SLOT_MODE_MONO : I2S_SLOT_MODE_STEREO));
            _pdm_rx_cfg.gpio_cfg = {
                .clk = static_cast<gpio_num_t>(_info.configs["clk_pin"].getValue<int>()),
                .din = static_cast<gpio_num_t>(_info.configs["din_pin"].getValue<int>()),
                .invert_flags = { .clk_inv = false },
            };

            auto ret = i2s_channel_init_pdm_rx_mode(_rx_chan, &_pdm_rx_cfg);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to initialize I2S channel in PDM RX mode: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to initialize I2S channel in PDM RX mode");
            }
        }
#else
        {
            const gpio_num_t bclk = static_cast<gpio_num_t>(_info.configs["bclk_pin"].getValue<int>());
            const gpio_num_t ws = static_cast<gpio_num_t>(_info.configs["ws_pin"].getValue<int>());
            const gpio_num_t din = BOARD_I2S_DIN_PIN;
            const gpio_num_t dout = BOARD_I2S_DOUT_PIN;
            const uint32_t sample_rate = static_cast<uint32_t>(_info.configs["sr"].getValue<int>());

            _std_cfg = {
                .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
                .slot_cfg = {
                    .data_bit_width = I2S_DATA_BIT_WIDTH_16BIT,
                    .slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT,
                    .slot_mode = I2S_SLOT_MODE_MONO,
                    .slot_mask = I2S_STD_SLOT_LEFT,
                    .ws_width = 32,
                    .ws_pol = false,
                    .bit_shift = true,
                    .left_align = true,
                    .big_endian = false,
                    .bit_order_lsb = false,
                },
                .gpio_cfg = {
                    .mclk = I2S_GPIO_UNUSED,
                    .bclk = bclk,
                    .ws   = ws,
                    .dout = dout,
                    .din  = din,
                    .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
                },
            };

            auto ret = i2s_channel_init_std_mode(_rx_chan, &_std_cfg);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to initialize I2S channel in STD mode: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to initialize I2S channel in STD mode");
            }
        }
#endif

        _frame_index = 0;
        _data_bytes_available = 0;
        {
            auto ret = i2s_channel_enable(_rx_chan);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to enable I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to enable I2S channel");
            }
        }

        _info.status = Status::Idle;

        return STATUS_OK();
    }

    core::Status deinit() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status == Status::Locked)
        {
            return STATUS(EBUSY, "Sensor is locked");
        }

        _data_buffer_capacity_frames = 0;
        _data_buffer_capacity_bytes = 0;

        _frame_index = 0;
        _data_bytes_available = 0;

        if (_data_buffer) [[likely]]
        {
            if (_data_buffer.use_count() > 1) [[unlikely]]
            {
                LOG(ERROR, "Data buffer is already in use by another process");
                return STATUS(EBUSY, "Data buffer is already in use");
            }

            _data_buffer.reset();
        }

        if (_rx_chan) [[likely]]
        {
            auto ret = i2s_channel_disable(_rx_chan);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to disable I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to disable I2S channel");
            }
            ret = i2s_del_channel(_rx_chan);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to delete I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to delete I2S channel");
            }
            _rx_chan = nullptr;
        }

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    core::Status updateConfig(const core::ConfigMap &configs) noexcept override
    {
        return STATUS(ENOTSUP, "Update config is not supported for LIS3DHTR sensor");
    }

    inline size_t dataAvailable() const noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return 0;
        }

        const std::lock_guard<std::mutex> lock(_lock);

        return internalDataAvailable();
    }

    inline size_t dataClear() noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return 0;
        }

        const std::lock_guard<std::mutex> lock(_lock);

        const size_t data_available = internalDataAvailable();
        if (data_available == 0) [[unlikely]]
        {
            return 0;
        }

        return internalDataDiscard(data_available);
    }

    inline core::Status readDataFrame(core::DataFrame<std::unique_ptr<core::Tensor>> &data_frame,
        size_t batch_size) noexcept override
    {
        if (batch_size == 0) [[unlikely]]
        {
            LOG(ERROR, "Batch size cannot be zero");
            return STATUS(EINVAL, "Batch size cannot be zero");
        }

        if (!initialized()) [[unlikely]]
        {
            LOG(ERROR, "Sensor is not initialized");
            return STATUS(ENXIO, "Sensor is not initialized");
        }
        if (_info.status != Status::Idle) [[unlikely]]
        {
            LOG(ERROR, "Sensor is not in idle state");
            return STATUS(EINVAL, "Sensor is not in idle state");
        }
        if (_data_buffer.use_count() > 1) [[unlikely]]
        {
            LOG(ERROR, "Data buffer is already in use by another process");
            return STATUS(EBUSY, "Data buffer is already in use");
        }

        if (batch_size > _data_buffer_capacity_frames) [[unlikely]]
        {
            LOG(WARNING, "Batch size exceeds buffer capacity: %zu > %zu", batch_size, _data_buffer_capacity_frames);
            batch_size = _data_buffer_capacity_frames;
        }

        const std::lock_guard<std::mutex> lock(_lock);

        _info.status = Status::Locked;

        data_frame.timestamp = std::chrono::steady_clock::now();

        size_t read = 0;
        {
            size_t size = batch_size * _channels * sizeof(int16_t);
            auto ret = i2s_channel_read(_rx_chan, _data_buffer.get(), size, &read,
                pdMS_TO_TICKS((_buffered_duration + 1) * 1000));
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to read data from I2]S channel: %s", esp_err_to_name(ret));
                _info.status = Status::Idle;
                return STATUS(EIO, "Failed to read data from I2S channel");
            }
            internalConsumeData(read);

            if (read != size) [[unlikely]]
            {
                LOG(WARNING, "Read %zu bytes, expected %zu bytes", read, size);
                if (!read) [[unlikely]]
                {
                    _info.status = Status::Idle;
                    return STATUS(EIO, "No data read from I2S channel");
                }
                batch_size = read / (_channels * sizeof(int16_t));
                read = batch_size * (_channels * sizeof(int16_t));
            }
        }
        data_frame.data = core::Tensor::create(core::Tensor::Type::Int16,
            core::Tensor::Shape(static_cast<int>(batch_size), static_cast<int>(_channels)), _data_buffer, read);

        data_frame.index = _frame_index;
        _frame_index += batch_size;

        _info.status = Status::Idle;

        return data_frame.data ? STATUS_OK() : STATUS(ENOMEM, "Failed to create data frame tensor");
    }

protected:
    static inline bool isrOnReceive(i2s_chan_handle_t, i2s_event_data_t *event, void *user_ctx)
    {
        if (!event || !user_ctx) [[unlikely]]
        {
            return false;
        }
        auto *self = static_cast<SensorI2SMic *>(user_ctx);

        size_t size = event->size;
        size_t data_bytes_available = self->_data_bytes_available.load(std::memory_order_acquire);
        while (1)
        {
            const size_t new_data_bytes_available
                = std::min(data_bytes_available + size, self->_data_buffer_capacity_bytes);
            if (self->_data_bytes_available.compare_exchange_strong(data_bytes_available, new_data_bytes_available,
                    std::memory_order_release, std::memory_order_relaxed)) [[likely]]
            {
                return false;
            }
        }

        return false;
    }

private:
    inline size_t internalDataAvailable() const noexcept
    {
        return _data_bytes_available.load(std::memory_order_acquire) / (_channels * sizeof(int16_t));
    }

    inline size_t internalDataDiscard(size_t size) noexcept
    {
        size_t discarded = 0;
        while (discarded < size)
        {
            size_t read = std::min(static_cast<size_t>((size - discarded) * _channels * sizeof(int16_t)),
                _data_buffer_capacity_bytes);
            auto ret = i2s_channel_read(_rx_chan, _data_buffer.get(), read, &read,
                pdMS_TO_TICKS((_buffered_duration + 1) * 1000));
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to read data from I2S channel: %s", esp_err_to_name(ret));
            }
            internalConsumeData(read);
            discarded += read;
        }
        return size;
    }

    inline void internalConsumeData(size_t size) noexcept
    {
        if (!size) [[unlikely]]
        {
            return;
        }
        size_t data_bytes_available = _data_bytes_available.load(std::memory_order_acquire);
        while (1)
        {
            size = std::min(size, data_bytes_available);
            const size_t new_data_bytes_available = data_bytes_available - size;
            if (_data_bytes_available.compare_exchange_strong(data_bytes_available, new_data_bytes_available,
                    std::memory_order_release, std::memory_order_relaxed)) [[likely]]
            {
                return;
            }
        }
    }

    mutable std::mutex _lock;
    static portMUX_TYPE _read_spinlock;

    std::shared_ptr<std::byte[]> _data_buffer = nullptr;
    size_t _data_buffer_capacity_frames = 0;
    size_t _data_buffer_capacity_bytes = 0;

    size_t _channels = 1;
    size_t _buffered_duration = 0;

    size_t _frame_index = 0;
    std::atomic<size_t> _data_bytes_available = 0;

    i2s_chan_handle_t _rx_chan = nullptr;
    i2s_chan_config_t _rx_chan_cfg = {};
    i2s_pdm_rx_config_t _pdm_rx_cfg = {};
    i2s_std_config_t _std_cfg = {};
    i2s_event_callbacks_t _event_cbs
        = { .on_recv = isrOnReceive, .on_recv_q_ovf = nullptr, .on_sent = nullptr, .on_send_q_ovf = nullptr };
};

} // namespace porting

#endif