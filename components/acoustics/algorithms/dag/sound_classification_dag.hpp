#pragma once
#ifndef SOUND_CLASSIFICATION_DAG_HPP
#define SOUND_CLASSIFICATION_DAG_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_dag.hpp"

#include <memory>
#include <string>

namespace algorithms { namespace node {
    class SpeechCommandsNode;
}} // namespace algorithms::node

namespace algorithms { namespace dag {

    class SoundClassificationDAG final: public module::MDAG
    {
    public:
        explicit SoundClassificationDAG(std::string_view name) noexcept : module::MDAG(name) { }

        std::shared_ptr<core::Tensor> getInputTensor(size_t index = 0) const noexcept override
        {
            auto input_node = node("ESPFeatureExtractorNode");
            if (input_node && !input_node->inputs().empty() && index < input_node->inputs().size())
            {
                auto mio = input_node->inputs()[index];
                if (mio && mio->operator()())
                {
                    return std::shared_ptr<core::Tensor>(mio->operator()(), [](core::Tensor *) { });
                }
            }
            return nullptr;
        }

        std::shared_ptr<core::Tensor> getOutputTensor(size_t index = 0) const noexcept override
        {
            auto output_node = node("SpeechCommandsNode");
            if (output_node && !output_node->outputs().empty() && index < output_node->outputs().size())
            {
                auto mio = output_node->outputs()[index];
                if (mio && mio->operator()())
                {
                    return std::shared_ptr<core::Tensor>(mio->operator()(), [](core::Tensor *) { });
                }
            }
            return nullptr;
        }
        size_t getActualOutputSize() const noexcept;
    };

    class SoundClassificationDAGBuilder
    {
    public:
        static std::shared_ptr<module::MDAG> create(const core::ConfigMap &configs);
        static constexpr const char *FEATURE_EXTRACTOR_NODE_NAME = "ESPFeatureExtractorNode";
        static constexpr const char *INFERENCE_NODE_NAME = "SpeechCommandsNode";

    private:
        static constexpr size_t AUDIO_SAMPLES = 44032;
        static constexpr size_t FEATURE_COUNT = 9976;

        static std::shared_ptr<core::Tensor> createInputTensor();

        static std::shared_ptr<core::Tensor> createFeatureTensor();

        static std::shared_ptr<core::Tensor> createOutputTensor(size_t output_size = 100);

        static std::shared_ptr<module::MNode> createFeatureExtractorNode(const core::ConfigMap &configs,
            std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio);

        static std::shared_ptr<module::MNode> createInferenceNode(const core::ConfigMap &configs,
            std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio);

        template<typename T>
        static std::shared_ptr<core::Tensor> allocateAlignedTensor(const core::Tensor::Shape &shape,
            core::Tensor::Type type);

        static size_t getOutputSize(const core::ConfigMap &configs);

        static core::Status validateConfig(const core::ConfigMap &configs);
    };

    std::shared_ptr<module::MDAG> createSoundClassificationDAG(const core::ConfigMap &configs);

}} // namespace algorithms::dag

#endif // SOUND_CLASSIFICATION_DAG_HPP
