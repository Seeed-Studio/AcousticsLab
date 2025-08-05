#pragma once
#ifndef ESP_FEATURE_EXTRACTOR_NODE_HPP
#define ESP_FEATURE_EXTRACTOR_NODE_HPP

#include "core/logger.hpp"
#include "core/status.hpp"
#include "feature_extractor_node.hpp"

#include "dl_rfft.h"
#include <esp_heap_caps.h>

#include <cstdint>
#include <memory>

namespace porting { namespace algorithms { namespace node {

    class ESPFeatureExtractorNode final: public FeatureExtractorNode
    {
    public:
        static constexpr size_t AUDIO_SAMPLES = 44032;
        static constexpr size_t FRAME_LEN = 2048;
        static constexpr size_t HOP_LEN = 1024;
        static constexpr size_t NUM_FRAMES = 43;
        static constexpr size_t FFT_SIZE = 2048;
        static constexpr size_t FEATURES_PER_FRAME = 232;
        static constexpr size_t OUTPUT_SIZE = NUM_FRAMES * FEATURES_PER_FRAME;

        ESPFeatureExtractorNode(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs,
            int priority);

        ~ESPFeatureExtractorNode() override;

        core::Status config(const core::ConfigMap &configs) noexcept override;

        size_t getInputSampleCount() const noexcept override
        {
            return AUDIO_SAMPLES;
        }
        size_t getOutputFeatureCount() const noexcept override
        {
            return OUTPUT_SIZE;
        }
        core::Tensor::Type getInputDataType() const noexcept override
        {
            return core::Tensor::Type::Int16;
        }
        core::Tensor::Type getOutputDataType() const noexcept override
        {
            return core::Tensor::Type::Float32;
        }

    protected:
        core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override;

    private:
        dl_fft_f32_t *_fft_handle;

        std::shared_ptr<float[]> _fft_input_buffer;
        std::shared_ptr<float[]> _log_features_buffer;
        std::shared_ptr<float[]> _audio_float_buffer;
        std::shared_ptr<float[]> _blackman_window;

        bool _initialized;

        core::Status initialize() noexcept;

        core::Status initBlackmanWindow() noexcept;

        void convertInt16ToFloat(const int16_t *input, float *output) noexcept;

        void prepareFrame(const float *audio_data, size_t frame_idx) noexcept;

        void normalizeGlobally(float *features, size_t total_features) noexcept;

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

    std::shared_ptr<module::MNode> createESPFeatureExtractorNode(const core::ConfigMap &configs, module::MIOS *inputs,
        module::MIOS *outputs, int priority);

}}} // namespace porting::algorithms::node

#endif // ESP_FEATURE_EXTRACTOR_NODE_HPP
