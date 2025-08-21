#pragma once
#ifndef SPEECH_COMMANDS_PRE_NODE_HPP
#define SPEECH_COMMANDS_PRE_NODE_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_node.hpp"

#include "dl_rfft.h"
#include <esp_heap_caps.h>

#include <cstdint>
#include <memory>

namespace porting { namespace algorithms { namespace node {

    class SpeechCommandsPreprocess final: public module::MNode
    {
    public:
        static constexpr size_t AUDIO_SAMPLES = 44032;
        static constexpr size_t FRAME_LEN = 2048;
        static constexpr size_t HOP_LEN = 1024;
        static constexpr size_t NUM_FRAMES = 43;
        static constexpr size_t FFT_SIZE = 2048;
        static constexpr size_t FEATURES_PER_FRAME = 232;
        static constexpr size_t OUTPUT_SIZE = NUM_FRAMES * FEATURES_PER_FRAME;

        inline SpeechCommandsPreprocess(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs,
            int priority)
            : module::MNode("SpeechCommandsPreprocess", std::move(inputs), std::move(outputs), priority),
              _fft_handle(nullptr), _fft_input_buffer(nullptr), _log_features_buffer(nullptr),
              _audio_float_buffer(nullptr), _blackman_window(nullptr), _initialized(false)
        {
            LOG(DEBUG, "Creating SpeechCommandsPreprocess with priority %d", priority);

            LOG(INFO, "SpeechCommandsPreprocess configured: input_samples=%zu, output_features=%zu", AUDIO_SAMPLES,
                OUTPUT_SIZE);
        }

        inline ~SpeechCommandsPreprocess() override
        {
            LOG(DEBUG, "Destroying SpeechCommandsPreprocess");

            if (_fft_handle)
            {
                dl_rfft_f32_deinit(_fft_handle);
                _fft_handle = nullptr;
            }
        }

        inline core::Status config(const core::ConfigMap &configs) noexcept override
        {
            LOG(DEBUG, "Reconfiguring SpeechCommandsPreprocess");

            return STATUS_OK();
        }

        size_t getInputSampleCount() const noexcept
        {
            return AUDIO_SAMPLES;
        }
        size_t getOutputFeatureCount() const noexcept
        {
            return OUTPUT_SIZE;
        }
        core::Tensor::Type getInputDataType() const noexcept
        {
            return core::Tensor::Type::Int16;
        }
        core::Tensor::Type getOutputDataType() const noexcept
        {
            return core::Tensor::Type::Float32;
        }

        inline core::Status initialize() noexcept
        {
            LOG(DEBUG, "Initializing SpeechCommandsPreprocess");

            _fft_input_buffer = allocateAligned<float>(FFT_SIZE);
            _log_features_buffer = allocateAligned<float>(OUTPUT_SIZE);
            _audio_float_buffer = allocateAligned<float>(AUDIO_SAMPLES);
            _blackman_window = allocateAligned<float>(FRAME_LEN);

            if (!_fft_input_buffer || !_log_features_buffer || !_audio_float_buffer || !_blackman_window)
            {
                LOG(ERROR, "Failed to allocate aligned buffers");
                return STATUS(ENOMEM, "Buffer allocation failed");
            }

            _fft_handle = dl_rfft_f32_init(FFT_SIZE, MALLOC_CAP_SPIRAM);
            if (!_fft_handle)
            {
                LOG(ERROR, "ESP-DL RFFT initialization failed");
                return STATUS(EFAULT, "ESP-DL RFFT initialization failed");
            }

            LOG(DEBUG, "Initializing Blackman window with %zu coefficients", FRAME_LEN);
            for (size_t n = 0; n < FRAME_LEN; ++n)
            {
                float n_norm = static_cast<float>(n) / static_cast<float>(FRAME_LEN);
                _blackman_window[n] = 0.42f - 0.5f * std::cos(2.0f * std::numbers::pi_v<float> * n_norm)
                                      + 0.08f * std::cos(4.0f * std::numbers::pi_v<float> * n_norm);
            }
            LOG(DEBUG, "Blackman window initialized successfully");

            _initialized = true;
            LOG(INFO, "SpeechCommandsPreprocess initialized successfully");
            return STATUS_OK();
        }

        inline core::Status validateTensors(const module::MIOS &inputs, const module::MIOS &outputs) const noexcept
        {
            if (inputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommandsPreprocess requires exactly 1 input tensor");
            }

            if (outputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommandsPreprocess requires exactly 1 output tensor");
            }

            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

            if (!input_tensor || !output_tensor)
            {
                return STATUS(EINVAL, "Input or output tensor is null");
            }

            if (input_tensor->dtype() != getInputDataType())
            {
                LOG(ERROR, "Input tensor data type mismatch: expected %d, got %d", static_cast<int>(getInputDataType()),
                    static_cast<int>(input_tensor->dtype()));
                return STATUS(EINVAL, "Input tensor data type mismatch");
            }

            if (static_cast<size_t>(input_tensor->shape().dot()) != getInputSampleCount())
            {
                LOG(ERROR, "Input tensor size mismatch: expected %zu, got %d", getInputSampleCount(),
                    input_tensor->shape().dot());
                return STATUS(EINVAL, "Input tensor size mismatch");
            }

            if (output_tensor->dtype() != getOutputDataType())
            {
                LOG(ERROR, "Output tensor data type mismatch: expected %d, got %d",
                    static_cast<int>(getOutputDataType()), static_cast<int>(output_tensor->dtype()));
                return STATUS(EINVAL, "Output tensor data type mismatch");
            }

            if (static_cast<size_t>(output_tensor->shape().dot()) != getOutputFeatureCount())
            {
                LOG(ERROR, "Output tensor size mismatch: expected %zu, got %d", getOutputFeatureCount(),
                    output_tensor->shape().dot());
                return STATUS(EINVAL, "Output tensor size mismatch");
            }

            return STATUS_OK();
        }

    protected:
        inline core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override
        {
            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

            const int16_t *raw_audio_data = input_tensor->data<int16_t>();
            float *output_features = output_tensor->data<float>();

            if (!raw_audio_data || !output_features)
            {
                return STATUS(EFAULT, "Failed to get tensor data pointers");
            }

            LOG(DEBUG, "Starting feature generation for %zu frames", NUM_FRAMES);

            convertInt16ToFloat(raw_audio_data, _audio_float_buffer.get());

            for (size_t frame_idx = 0; frame_idx < NUM_FRAMES; ++frame_idx)
            {
                prepareFrame(_audio_float_buffer.get(), frame_idx);

                dl_rfft_f32_run(_fft_handle, _fft_input_buffer.get());

                float *current_log_features = _log_features_buffer.get() + (frame_idx * FEATURES_PER_FRAME);
                for (size_t j = 0; j < FEATURES_PER_FRAME; ++j)
                {
                    float real = _fft_input_buffer[j * 2 + 0];
                    float imag = _fft_input_buffer[j * 2 + 1];
                    float mag = std::sqrtf(real * real + imag * imag);
                    current_log_features[j] = std::logf(
                        mag < std::numeric_limits<float>::epsilon() ? std::numeric_limits<float>::epsilon() : mag);
                }
            }

            normalizeGlobally(_log_features_buffer.get(), OUTPUT_SIZE);

            std::memcpy(output_features, _log_features_buffer.get(), OUTPUT_SIZE * sizeof(float));

            LOG(DEBUG, "Feature generation completed, output size: %zu", OUTPUT_SIZE);
            return STATUS_OK();
        }

    private:
        dl_fft_f32_t *_fft_handle;

        std::shared_ptr<float[]> _fft_input_buffer;
        std::shared_ptr<float[]> _log_features_buffer;
        std::shared_ptr<float[]> _audio_float_buffer;
        std::shared_ptr<float[]> _blackman_window;

        bool _initialized;

        inline void convertInt16ToFloat(const int16_t *input, float *output) noexcept
        {
            for (size_t i = 0; i < AUDIO_SAMPLES; ++i)
            {
                output[i] = static_cast<float>(input[i]) / 32768.0f;
            }
        }

        inline void prepareFrame(const float *audio_data, size_t frame_idx) noexcept
        {
            size_t start_in_padded = frame_idx * HOP_LEN;
            size_t padding_size = HOP_LEN;

            for (size_t i = 0; i < HOP_LEN; ++i)
            {
                size_t pos_in_padded = start_in_padded + i;
                if (pos_in_padded < padding_size)
                {
                    _fft_input_buffer[i] = 0.0f;
                }
                else
                {
                    _fft_input_buffer[i] = audio_data[pos_in_padded - padding_size];
                }
            }

            std::memcpy(_fft_input_buffer.get() + HOP_LEN, audio_data + (frame_idx * HOP_LEN), HOP_LEN * sizeof(float));

            for (size_t i = 0; i < FRAME_LEN; ++i)
            {
                _fft_input_buffer[i] *= _blackman_window[i];
            }
        }

        inline void normalizeGlobally(float *features, size_t total_features) noexcept
        {
            double mean = 0.0;
            for (size_t i = 0; i < total_features; ++i)
            {
                mean += features[i];
            }
            mean /= total_features;

            double variance = 0.0;
            for (size_t i = 0; i < total_features; ++i)
            {
                float diff = features[i] - mean;
                variance += diff * diff;
            }
            variance /= total_features;
            float std_dev = std::sqrtf(variance);

            const float epsilon = std::numeric_limits<float>::epsilon();
            float inv_std = 1.0f / (std_dev + epsilon);

            for (size_t i = 0; i < total_features; ++i)
            {
                features[i] = (features[i] - static_cast<float>(mean)) * inv_std;
            }
        }

        template<typename T>
        std::shared_ptr<T[]> allocateAligned(size_t count) noexcept
        {
            void *ptr = heap_caps_aligned_alloc(16, count * sizeof(T), MALLOC_CAP_SPIRAM);
            if (!ptr)
            {
                return nullptr;
            }

            return std::shared_ptr<T[]>(static_cast<T *>(ptr), [](T *p) {
                if (p)
                {
                    heap_caps_free(p);
                }
            });
        }
    };

    inline std::shared_ptr<module::MNode> createSpeechCommandsPreprocess(const core::ConfigMap &configs,
        module::MIOS *inputs, module::MIOS *outputs, int priority)
    {
        LOG(DEBUG, "Creating SpeechCommandsPreprocess via builder function");

        module::MIOS input_mios = inputs ? *inputs : module::MIOS {};
        module::MIOS output_mios = outputs ? *outputs : module::MIOS {};

        auto node = std::make_shared<SpeechCommandsPreprocess>(configs, std::move(input_mios), std::move(output_mios),
            priority);

        auto init_status = node->initialize();
        if (!init_status)
        {
            LOG(ERROR, "Failed to initialize SpeechCommandsPreprocess: %s", init_status.message().c_str());
            return nullptr;
        }

        module::MIOS validate_inputs = inputs ? *inputs : module::MIOS {};
        module::MIOS validate_outputs = outputs ? *outputs : module::MIOS {};
        auto validate_status = node->validateTensors(validate_inputs, validate_outputs);
        if (!validate_status)
        {
            LOG(ERROR, "SpeechCommandsPreprocess tensor validation failed: %s", validate_status.message().c_str());
            return nullptr;
        }

        return node;
    }

}}} // namespace porting::algorithms::node

#endif // SPEECH_COMMANDS_PRE_NODE_HPP
