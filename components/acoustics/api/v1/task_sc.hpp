#pragma once
#ifndef TASK_SC_HPP
#define TASK_SC_HPP

#include "common.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/encoder/adpcm.hpp"
#include "core/encoder/ascii.hpp"
#include "core/encoder/opus.hpp"
#include "core/model.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "core/types.hpp"

#include "hal/device.hpp"
#include "hal/engine.hpp"
#include "hal/sensor.hpp"

#include "module/module_dag.hpp"
#include "module/module_node.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>

namespace v1 {

namespace shared {

    inline std::atomic<bool> is_sampling = false;
    inline std::atomic<bool> is_invoking = false;

    inline constexpr const size_t buffer_size = 131072;
    inline constexpr const size_t buffer_size_mask = buffer_size - 1;
    inline std::mutex buffer_mutex;
    inline size_t buffer_head = 0;
    inline size_t buffer_tail = 0;
    inline std::vector<int16_t> buffer(buffer_size);
    inline std::chrono::steady_clock::time_point buffer_base_ts = std::chrono::steady_clock::now();

    inline constexpr const size_t sample_chunk_ms = 100;
    inline constexpr const size_t invoke_pull_ms = 10;

    inline constexpr const float overlap_ratio_min = 0.f;
    inline constexpr const float overlap_ratio_max = 0.75f;
    inline std::atomic<float> overlap_ratio = 0.5f;
    inline std::atomic<bool> rms_normalize = false;
    inline constexpr const float threshold_min = 0.f;
    inline constexpr const float threshold_max = 1.f;
    inline std::atomic<float> threshold = 0.0f;
} // namespace shared

struct TaskSC final
{
    class Sample final: public api::Task
    {
    public:
        Sample(api::Context &context, hal::Transport &transport, size_t id, hal::Sensor *sensor, std::string tag,
            const volatile size_t &external_task_id) noexcept
            : api::Task(context, transport, id, v1::defaults::task_priority), _sensor(sensor), _tag(std::move(tag)),
              _external_task_id(external_task_id), _internal_task_id(_external_task_id),
              _start_time(std::chrono::steady_clock::now()), _current_id(0), _current_id_next(0), _sr(0), _fs(0)
        {
            {
                auto device = hal::DeviceRegistry::getDevice();
                while (_external_task_id != _internal_task_id)
                {
                    bool expected = false;
                    if (shared::is_sampling.compare_exchange_strong(expected, true))
                        break;
                    if (device)
                        device->sleep(5);
                }
            }

            if (!_sensor) [[unlikely]]
            {
                LOG(ERROR, "Sensor is null");
                return;
            }

            {
                auto it = _sensor->info().configs.find("sr");
                if (it != _sensor->info().configs.end())
                {
                    auto sr = it->second.getValue<int>();
                    if (sr <= 0)
                    {
                        LOG(ERROR, "Invalid sample rate: %d", sr);
                        return;
                    }
                    _sr = sr;
                    _fs = _sr * shared::sample_chunk_ms / 1000;
                    LOG(INFO, "Sample rate: %d, Frame size: %zu", _sr, _fs);
                }
            }

            {
                auto adpcm_size = core::EncoderADPCMIMA::estimate(_fs);
                auto opus_size = core::EncoderLIBOPUS::estimate(_fs, _sr);
                LOG(INFO, "Estimated ADPCM size: %zu, OPUS size: %zu", adpcm_size, opus_size);

                _adpcm_encoder = core::EncoderADPCMIMA::create(nullptr, adpcm_size);
                _opus_encoder = core::EncoderLIBOPUS::create(nullptr, opus_size);

                auto base64_size = core::EncoderASCIIBase64::estimate(std::max(adpcm_size, opus_size));
                LOG(INFO, "Estimated Base64 size: %zu", base64_size);

                _base64_encoder = core::EncoderASCIIBase64::create(nullptr, base64_size);
            }

            {
                _sensor->dataClear();
            }

            {
                std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                v1::shared::buffer_head = 0;
                v1::shared::buffer_tail = 0;
                v1::shared::buffer_base_ts = _start_time;
                _current_id = v1::shared::buffer_tail;
            }
        }

        ~Sample() noexcept override
        {
            shared::is_sampling = false;
        }

        inline core::Status operator()(api::Executor &executor) override
        {
            shared::is_sampling = true;

            if (_external_task_id != _internal_task_id) [[unlikely]]
            {
                return STATUS_OK();
            }
            if (!_sensor || !_sensor->initialized()) [[unlikely]]
            {
                return replyWithStatus(STATUS(EINVAL, "Microphone sensor not initialized"),
                    std::chrono::steady_clock::now());
            }

            const size_t available = _sensor->dataAvailable();
            if (available < _fs)
            {
                return executor.submit(getptr(), getNextDataDelay(available));
            }

            auto status = STATUS_OK();
            auto df_ts = std::chrono::steady_clock::now();
            void (*encoder_writer)(Sample &, core::ObjectWriter &) = nullptr;
            {
                core::DataFrame<std::unique_ptr<core::Tensor>> data_frame;
                auto status = _sensor->readDataFrame(data_frame, _fs);
                if (!status) [[unlikely]]
                {
                    return replyWithStatus(status, std::chrono::steady_clock::now());
                }
                if (!data_frame.data) [[unlikely]]
                {
                    LOG(ERROR, "Data frame is null");
                    return replyWithStatus(STATUS(EFAULT, "Data frame is null"), std::chrono::steady_clock::now());
                }
                const auto &shape = data_frame.data->shape();
                if (shape.size() != 2 || shape[0] < _fs || shape[1] != 1
                    || data_frame.data->dtype() != core::Tensor::Type::Int16) [[unlikely]]
                {
                    LOG(ERROR, "Data frame shape or type mismatch: shape=(%zu,%zu), dtype=%d", shape[0], shape[1],
                        static_cast<int>(data_frame.data->dtype()));
                    return replyWithStatus(STATUS(EINVAL, "Data frame shape or type mismatch"),
                        std::chrono::steady_clock::now());
                }

                df_ts = data_frame.timestamp;
                const auto size = shape[0];
                {
                    const std::lock_guard<std::mutex> lock(v1::shared::buffer_mutex);
                    const size_t head = v1::shared::buffer_head;
                    const size_t tail = v1::shared::buffer_tail;
                    const size_t next_head = head + size;

                    if (static_cast<size_t>(next_head - tail) > v1::shared::buffer_size) [[unlikely]]
                    {
                        v1::shared::buffer_tail = next_head - v1::shared::buffer_size;
                    }
                    for (size_t i = 0; i < size; ++i)
                    {
                        v1::shared::buffer[(head + i) & v1::shared::buffer_size_mask]
                            = data_frame.data->data<int16_t>()[i];
                    }
                    v1::shared::buffer_head = next_head;
                }

                _current_id_next = _current_id + 1;
                if (shared::is_invoking)
                {
                    status = encodeADPCM(data_frame.data->data<int16_t>(), size);
                    encoder_writer = writeADPCM;
                }
                else
                {
                    status = encodeOPUS(data_frame.data->data<int16_t>(), size);
                    encoder_writer = writeOPUS;
                }
            }

            status = replyWithStatus(status, df_ts, encoder_writer);
            if (!status) [[unlikely]]
            {
                return status;
            }

            return executor.submit(getptr(), getNextDataDelay(_sensor->dataAvailable()));
        }

    private:
        core::Status replyWithStatus(core::Status status, std::chrono::steady_clock::time_point current,
            void (*encoder_writer)(Sample &, core::ObjectWriter &) = nullptr) noexcept
        {
            auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
                std::bind(&v1::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v1::defaults::flush_callback, std::ref(_transport)));

            writer["type"] += v1::ResponseType::Stream;
            writer["name"] += "PCMData";
            writer["tag"] << _tag;
            writer["code"] += status.code();
            writer["msg"] << status.message();
            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                data["id"] += _current_id;
                data["ts"] += std::chrono::duration_cast<std::chrono::milliseconds>(current - _start_time).count();
                if (encoder_writer)
                {
                    encoder_writer(*this, data);
                }
            }
            _current_id = _current_id_next;

            return status;
        }

        core::Status encodeADPCM(const int16_t *data, size_t size) noexcept
        {
            if (!_adpcm_encoder || !_base64_encoder) [[unlikely]]
            {
                return STATUS(EINVAL, "ADPCM encoder or Base64 encoder not initialized");
            }

            _adpcm_state = _adpcm_encoder->state();
            _adpcm_encoder->encode(data, size, [&](const void *encoded, size_t encoded_size) {
                _base64_encoder->encode(static_cast<const uint8_t *>(encoded), encoded_size,
                    [&](const void *b64, size_t b64_size) {
                        _b64_str = std::string_view(static_cast<const char *>(b64), b64_size);
                        return 0;
                    });
                return _base64_encoder->error();
            });
            auto ret = _adpcm_encoder->error();
            if (ret) [[unlikely]]
            {
                return STATUS(ret, "ADPCM encoding failed");
            }
            return STATUS_OK();
        }

        static void writeADPCM(Sample &t, core::ObjectWriter &writer) noexcept
        {
            writer["ch"] += 1;
            writer["sr"] += static_cast<int>(t._sr);
            writer["bw"] += 16;
            writer["pd"] += t._adpcm_state.predictor;
            writer["si"] += t._adpcm_state.step_index;
            writer["adpcm"] << t._b64_str;
        }

        core::Status encodeOPUS(const int16_t *data, size_t size) noexcept
        {
            if (!_opus_encoder || !_base64_encoder) [[unlikely]]
            {
                return STATUS(EINVAL, "OPUS encoder or Base64 encoder not initialized");
            }

            _opus_state = _opus_encoder->state();
            _opus_encoder->encode(data, size, [&](const void *encoded, size_t encoded_size) {
                _base64_encoder->encode(static_cast<const uint8_t *>(encoded), encoded_size,
                    [&](const void *b64, size_t b64_size) {
                        _b64_str = std::string_view(static_cast<const char *>(b64), b64_size);
                        return 0;
                    });
                return _base64_encoder->error();
            });
            auto ret = _opus_encoder->error();
            if (ret) [[unlikely]]
            {
                return STATUS(ret, "OPUS encoding failed");
            }
            return STATUS_OK();
        }

        static void writeOPUS(Sample &t, core::ObjectWriter &writer) noexcept
        {
            writer["ch"] += 1;
            writer["sr"] += static_cast<int>(t._opus_state.sample_rate);
            writer["bw"] += 16;
            writer["opus"] << t._b64_str;
        }

        inline size_t getNextDataDelay(size_t current_available) const noexcept
        {
            if (current_available >= _fs) [[unlikely]]
            {
                return 0;
            }
            return (_fs - current_available) * 1000 / (_sr + 1);
        }

        hal::Sensor *_sensor;
        const std::string _tag;
        const volatile size_t &_external_task_id;
        const size_t _internal_task_id;
        const std::chrono::steady_clock::time_point _start_time;
        size_t _current_id;
        size_t _current_id_next;
        size_t _sr;
        size_t _fs;

        std::unique_ptr<core::EncoderADPCMIMA> _adpcm_encoder = nullptr;
        std::unique_ptr<core::EncoderLIBOPUS> _opus_encoder = nullptr;
        std::unique_ptr<core::EncoderASCIIBase64> _base64_encoder = nullptr;

        core::EncoderADPCMIMA::State _adpcm_state = {};
        core::EncoderLIBOPUS::State _opus_state = {};
        std::string_view _b64_str = {};
    };

    class Invoke final: public api::Task
    {
    public:
        Invoke(api::Context &context, hal::Transport &transport, size_t id, std::shared_ptr<module::MDAG> dag,
            std::string tag, const volatile size_t &external_task_id) noexcept
            : api::Task(context, transport, id, v1::defaults::task_priority), _dag(std::move(dag)),
              _tag(std::move(tag)), _external_task_id(external_task_id), _internal_task_id(_external_task_id),
              _start_time(std::chrono::steady_clock::now()), _rms_normalize(shared::rms_normalize.load()),
              _current_id(0), _current_id_next(0)
        {
            {
                auto device = hal::DeviceRegistry::getDevice();
                while (_external_task_id != _internal_task_id)
                {
                    bool expected = false;
                    if (shared::is_invoking.compare_exchange_strong(expected, true))
                        break;
                    if (device)
                        device->sleep(5);
                }
            }

            if (!_dag)
            {
                return;
            }
            {
                configureRMSNormalize(true);
            }
            {
                auto p_inp_node = _dag->node("input");
                if (!p_inp_node)
                    return;
                const auto &r_inp_mio = p_inp_node->input(0);
                if (!r_inp_mio)
                    return;
                auto p_inp_tsr = r_inp_mio->operator()();
                if (!p_inp_tsr || !p_inp_tsr->data() || p_inp_tsr->dtype() != core::Tensor::Type::Int16)
                    return;
                _input = p_inp_tsr;
            }
            {
                auto p_out_node = _dag->node("output");
                if (!p_out_node)
                    return;
                const auto &r_out_mio = p_out_node->output(0);
                if (!r_out_mio)
                    return;
                auto p_out_tsr = r_out_mio->operator()();
                if (!p_out_tsr || !p_out_tsr->data() || p_out_tsr->dtype() != core::Tensor::Type::Class)
                    return;
                _output = p_out_tsr;
            }
        }

        ~Invoke() noexcept override
        {
            shared::is_invoking = false;
        }

        inline core::Status operator()(api::Executor &executor) override
        {
            shared::is_invoking = true;

            if (_external_task_id != _internal_task_id) [[unlikely]]
            {
                return STATUS_OK();
            }
            if (!shared::is_sampling.load()) [[unlikely]]
            {
                return replyWithStatus(STATUS(EINVAL, "Sampling is not active"));
            }
            if (!_dag || !_input || !_output) [[unlikely]]
            {
                return replyWithStatus(STATUS(EFAULT, "DAG, input, or output is null"));
            }

            configureRMSNormalize();

            const auto &inp_shape = _input->shape();
            if (inp_shape.size() != 2 || inp_shape[0] > shared::buffer_size || inp_shape[1] != 1)
            {
                return replyWithStatus(STATUS(EINVAL, "Invalid input tensor shape"));
            }

            const size_t required = inp_shape[0];
            const size_t to_discard = static_cast<size_t>(std::round(required * (1.f - shared::overlap_ratio.load())));
            size_t available = 0;
            {
                const std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                const auto head = shared::buffer_head;
                const auto tail = shared::buffer_tail;
                available = head - tail;
                if (available < required) [[unlikely]]
                {
                    return executor.submit(getptr(), shared::invoke_pull_ms);
                }
                for (size_t i = 0; i < required; ++i)
                {
                    _input->data<int16_t>()[i] = shared::buffer[(tail + i) & shared::buffer_size_mask];
                }
                shared::buffer_tail = tail + std::min(to_discard, available);
            }
            _current_id_next = _current_id + 1;

            const auto s = std::chrono::steady_clock::now();
            auto status = _dag->operator()();
            const auto e = std::chrono::steady_clock::now();
            _perf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
            if (!status) [[unlikely]]
            {
                return replyWithStatus(status);
            }

            status = replyWithStatus(status);
            if (!status) [[unlikely]]
            {
                return status;
            }

            return executor.submit(getptr(), shared::invoke_pull_ms);
        }

    private:
        core::Status replyWithStatus(core::Status status) noexcept
        {
            auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
                std::bind(&v1::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v1::defaults::flush_callback, std::ref(_transport)));

            writer["type"] += v1::ResponseType::Stream;
            writer["name"] += "ClassifyResult";
            writer["tag"] += _tag;
            writer["code"] += status.code();
            writer["msg"] << status.message();

            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                data["id"] += _current_id;
                data["ts"] += std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - _start_time)
                                  .count();
                bool all_below_threshold = false;
                if (status && _output) [[likely]]
                {
                    auto cls_data = data["data"].writer<core::ArrayWriter>();
                    const int size = _output->shape().size() ? _output->shape()[0] : 0;
                    const auto classes = _output->data<core::class_t>();
                    all_below_threshold = std::all_of(classes, classes + size,
                        [](const core::class_t &c) { return c.confidence < shared::threshold.load(); });
                    if (!all_below_threshold)
                    {
                        for (int i = 0; i < size; ++i)
                        {
                            auto cls = cls_data.writer<core::ArrayWriter>();
                            const int id = classes[i].id;
                            const float confidence = classes[i].confidence;
                            cls += id;
                            cls << confidence;
                        }
                    }
                }
                data["allBelowThreshold"] += all_below_threshold;
                data["perfMs"] += _perf_ms;
            }
            _current_id = _current_id_next;

            return status;
        }

        void configureRMSNormalize(bool force = false) noexcept
        {
            if (!force)
            {
                const bool target = shared::rms_normalize.load();
                if (_rms_normalize == target)
                    return;
                _rms_normalize = target;
            }

            auto dag = _dag;
            if (!dag)
                return;
            auto p_node = dag->node("SpeechCommandsPreprocess");
            if (!p_node)
                return;
            auto status = p_node->config(core::ConfigMap {
                { "rms_normalize", _rms_normalize },
            });
            LOG(DEBUG, "Configured RMS normalize to %d: %s", _rms_normalize, status.message().c_str());
        }

        std::shared_ptr<module::MDAG> _dag;
        const std::string _tag;
        const volatile size_t &_external_task_id;
        const size_t _internal_task_id;
        const std::chrono::steady_clock::time_point _start_time;
        bool _rms_normalize;
        size_t _current_id;
        size_t _current_id_next;

        core::Tensor *_input = nullptr;
        core::Tensor *_output = nullptr;

        size_t _perf_ms = 0;
    };
};

} // namespace v1

#endif
