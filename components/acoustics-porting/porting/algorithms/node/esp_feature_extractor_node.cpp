#include "esp_feature_extractor_node.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace porting { namespace algorithms { namespace node {

    static const float PI = 3.14159265358979323846264338327950288419716939937510582097494459231f;

    ESPFeatureExtractorNode::ESPFeatureExtractorNode(const core::ConfigMap &configs, module::MIOS inputs,
        module::MIOS outputs, int priority)
        : FeatureExtractorNode("ESPFeatureExtractorNode", std::move(inputs), std::move(outputs), priority),
          _fft_handle(nullptr), _fft_input_buffer(nullptr), _log_features_buffer(nullptr), _audio_float_buffer(nullptr),
          _blackman_window(nullptr), _initialized(false)
    {
        LOG(DEBUG, "Creating ESPFeatureExtractorNode with priority %d", priority);

        LOG(INFO, "ESPFeatureExtractorNode configured: input_samples=%zu, output_features=%zu", AUDIO_SAMPLES,
            OUTPUT_SIZE);
    }

    ESPFeatureExtractorNode::~ESPFeatureExtractorNode()
    {
        LOG(DEBUG, "Destroying ESPFeatureExtractorNode");

        if (_fft_handle)
        {
            dl_rfft_f32_deinit(_fft_handle);
            _fft_handle = nullptr;
        }
    }

    core::Status ESPFeatureExtractorNode::config(const core::ConfigMap &configs) noexcept
    {
        LOG(DEBUG, "Reconfiguring ESPFeatureExtractorNode");

        return STATUS_OK();
    }

    core::Status ESPFeatureExtractorNode::forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept
    {
        if (!_initialized)
        {
            auto status = initialize();
            if (!status)
            {
                return status;
            }
        }

        auto status = validateTensors(inputs, outputs);
        if (!status)
        {
            return status;
        }

        auto input_tensor = inputs[0]->operator()();
        auto output_tensor = outputs[0]->operator()();

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
                float mag = sqrtf(real * real + imag * imag);
                current_log_features[j]
                    = logf(mag < std::numeric_limits<float>::epsilon() ? std::numeric_limits<float>::epsilon() : mag);
            }
        }

        normalizeGlobally(_log_features_buffer.get(), OUTPUT_SIZE);

        std::memcpy(output_features, _log_features_buffer.get(), OUTPUT_SIZE * sizeof(float));

        LOG(DEBUG, "Feature generation completed, output size: %zu", OUTPUT_SIZE);
        return STATUS_OK();
    }

    core::Status ESPFeatureExtractorNode::initialize() noexcept
    {
        LOG(DEBUG, "Initializing ESPFeatureExtractorNode");

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

        auto status = initBlackmanWindow();
        if (!status)
        {
            return status;
        }

        _initialized = true;
        LOG(INFO, "ESPFeatureExtractorNode initialized successfully");
        return STATUS_OK();
    }

    core::Status ESPFeatureExtractorNode::initBlackmanWindow() noexcept
    {
        LOG(DEBUG, "Initializing Blackman window with %zu coefficients", FRAME_LEN);

        for (size_t n = 0; n < FRAME_LEN; ++n)
        {
            float n_norm = static_cast<float>(n) / static_cast<float>(FRAME_LEN);
            _blackman_window[n] = 0.42f - 0.5f * cos(2.0f * PI * n_norm) + 0.08f * cos(4.0f * PI * n_norm);
        }

        LOG(DEBUG, "Blackman window initialized successfully");
        return STATUS_OK();
    }

    void ESPFeatureExtractorNode::convertInt16ToFloat(const int16_t *input, float *output) noexcept
    {
        for (size_t i = 0; i < AUDIO_SAMPLES; ++i)
        {
            output[i] = static_cast<float>(input[i]) / 32768.0f;
        }
    }

    void ESPFeatureExtractorNode::prepareFrame(const float *audio_data, size_t frame_idx) noexcept
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

    void ESPFeatureExtractorNode::normalizeGlobally(float *features, size_t total_features) noexcept
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
        float std_dev = sqrtf(variance);

        const float epsilon = std::numeric_limits<float>::epsilon();
        float inv_std = 1.0f / (std_dev + epsilon);

        for (size_t i = 0; i < total_features; ++i)
        {
            features[i] = (features[i] - static_cast<float>(mean)) * inv_std;
        }
    }

    std::shared_ptr<module::MNode> createESPFeatureExtractorNode(const core::ConfigMap &configs, module::MIOS *inputs,
        module::MIOS *outputs, int priority)
    {
        LOG(DEBUG, "Creating ESPFeatureExtractorNode via builder function");

        module::MIOS input_mios = inputs ? *inputs : module::MIOS {};
        module::MIOS output_mios = outputs ? *outputs : module::MIOS {};

        return std::make_shared<ESPFeatureExtractorNode>(configs, std::move(input_mios), std::move(output_mios),
            priority);
    }

}}} // namespace porting::algorithms::node
