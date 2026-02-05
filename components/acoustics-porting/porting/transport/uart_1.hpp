#pragma once
#ifndef UART_1_HPP
#define UART_1_HPP

#include "core/ring_buffer.hpp"
#include "hal/transport.hpp"

#include <driver/gpio.h>
#include <driver/uart.h>
#include <esp_err.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

namespace porting {

class TransportUART1 final: public hal::Transport
{
public:
    static inline core::ConfigObjectMap DEFAULT_CONFIGS() noexcept
    {
        return {};
    }

    TransportUART1() noexcept : Transport(Info(1, "UART 1", Type::UART, { DEFAULT_CONFIGS() })) { }

    core::Status init() noexcept override
    {
        if (_info.status != Status::Uninitialized)
        {
            return STATUS(ENXIO, "Transport is already initialized or in an invalid state");
        }

        auto ret = uart_driver_install(UART_NUM_1, _rx_buffer_size, _tx_buffer_size, 0, NULL, 0);
        if (ret != ESP_OK)
        {
            LOG(ERROR, "Failed to install UART driver: %s", esp_err_to_name(ret));
            return STATUS(EIO, "Failed to install UART driver");
        }

        ret = uart_param_config(UART_NUM_1, &_config);
        if (ret != ESP_OK)
        {
            LOG(ERROR, "Failed to configure UART parameters: %s", esp_err_to_name(ret));
            uart_driver_delete(UART_NUM_1);
            return STATUS(EIO, "Failed to configure UART parameters");
        }

        ret = uart_set_pin(UART_NUM_1, GPIO_NUM_43, GPIO_NUM_44, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
        if (ret != ESP_OK)
        {
            LOG(ERROR, "Failed to set UART pins: %s", esp_err_to_name(ret));
            uart_driver_delete(UART_NUM_1);
            return STATUS(EIO, "Failed to set UART pins");
        }

        if (!_rx_buffer)
        {
            _rx_buffer = core::RingBuffer<std::byte>::create(_rx_buffer_size);
            if (!_rx_buffer)
            {
                LOG(ERROR, "Failed to create RX buffer");
                return STATUS(ENOMEM, "Failed to create RX buffer");
            }
        }

        _info.status = Status::Idle;

        return STATUS_OK();
    }

    core::Status deinit() noexcept override
    {
        if (_info.status == Status::Locked)
        {
            return STATUS(EBUSY, "Transport is currently locked");
        }

        auto ret = uart_driver_delete(UART_NUM_1);
        if (ret != ESP_OK)
        {
            LOG(ERROR, "Failed to uninstall UART driver: %s", esp_err_to_name(ret));
            return STATUS(EIO, "Failed to uninstall UART driver");
        }

        if (_rx_buffer)
        {
            _rx_buffer.reset();
        }

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    core::Status updateConfig(const core::ConfigMap &configs) noexcept override
    {
        return STATUS(ENOTSUP, "Update config is not supported for UART transport");
    }

    inline size_t available() const noexcept override
    {
        if (!initialized() || !_rx_buffer) [[unlikely]]
        {
            return 0;
        }

        int ret = syncReadBuffer();
        if (ret != 0) [[unlikely]]
        {
            LOG(ERROR, "Failed to sync read buffer: %s", std::strerror(ret));
        }
        return _rx_buffer->size();
    }

    inline size_t read(void *data, size_t size) noexcept override
    {
        if (!_rx_buffer) [[unlikely]]
        {
            return 0;
        }

        int ret = syncReadBuffer();
        if (ret != 0) [[unlikely]]
        {
            LOG(ERROR, "Failed to sync read buffer: %s", std::strerror(ret));
            return 0;
        }

        return _rx_buffer->read(reinterpret_cast<std::byte *>(data), size);
    }

    inline size_t write(const void *data, size_t size) noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return size; // Not initialized, return size as if all data was written
        }
        if (!data || size == 0) [[unlikely]]
        {
            return 0;
        }

        size_t written = 0;
        while (written < size)
        {
            size_t to_write = std::min(size - written, static_cast<size_t>(_tx_buffer_size));
            int ret = uart_write_bytes(UART_NUM_1, static_cast<const unsigned char *>(data) + written, to_write);
            if (ret < 0) [[unlikely]]
            {
                LOG(ERROR, "Failed to write data to UART: %s", esp_err_to_name(ret));
                return -EIO;
            }
            written += ret;
        }

        return written;
    }

    inline int flush() noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return 0; // Not initialized, nothing to flush
        }
        return uart_wait_tx_done(UART_NUM_1, portMAX_DELAY);
    }

protected:
    inline int syncReadBuffer() const noexcept override
    {
        std::byte r_buf[32];
        for (int r_len = 0;;)
        {
            r_len = uart_read_bytes(UART_NUM_1, r_buf, sizeof(r_buf), pdMS_TO_TICKS(10));
            if (r_len > 0) [[likely]]
            {
                auto written = _rx_buffer->write(r_buf, r_len);
                if (written < r_len) [[unlikely]]
                {
                    LOG(ERROR, "Failed to write all read bytes to RX buffer, written: %zu, expected: %d", written,
                        r_len);
                }
            }
            else if (r_len == 0)
            {
                break;
            }
            else
            {
                return -EIO;
            }
        }

        return 0;
    }

    inline core::RingBuffer<std::byte> &getReadBuffer() noexcept override
    {
        return *_rx_buffer;
    }

private:
    static const inline size_t _tx_buffer_size = 32 * 1024;
    static const inline size_t _rx_buffer_size = 4096;
    static const inline uart_config_t _config = { .baud_rate = 921600,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
        .flags = { .allow_pd = 0, .backup_before_sleep = 0 } };

    std::unique_ptr<core::RingBuffer<std::byte>> _rx_buffer = nullptr;
};

} // namespace porting

#endif // UART_HPP
