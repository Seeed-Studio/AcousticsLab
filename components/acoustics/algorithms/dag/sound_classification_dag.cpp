#include "sound_classification_dag.hpp"
#include "../node/speech_commands_node.hpp"
#include "core/logger.hpp"
#include "hal/engine.hpp"
#include "module/module_dag.hpp"
#include "module/module_node.hpp"

#include <algorithm>
#include <esp_heap_caps.h>
#include <string>

namespace algorithms { namespace dag {

    std::shared_ptr<module::MDAG> SoundClassificationDAGBuilder::create(const core::ConfigMap &configs)
    {
        LOG(INFO, "Creating SoundClassificationDAG");

        auto status = validateConfig(configs);
        if (!status)
        {
            LOG(ERROR, "Invalid DAG configuration: %s", status.message().c_str());
            return nullptr;
        }

        auto dag = std::make_shared<SoundClassificationDAG>("SoundClassificationDAG");

        size_t output_size = getOutputSize(configs);
        if (output_size == 0)
        {
            LOG(ERROR, "Failed to determine model output size - cannot create DAG");
            return nullptr;
        }
        LOG(INFO, "DAG will be created with output size: %zu classes", output_size);

        auto input_tensor = createInputTensor();
        auto feature_tensor = createFeatureTensor();
        auto output_tensor = createOutputTensor(output_size);

        if (!input_tensor || !feature_tensor || !output_tensor)
        {
            LOG(ERROR, "Failed to create DAG tensors");
            return nullptr;
        }

        auto input_mio = std::make_shared<module::MIO>(input_tensor);
        auto feature_mio = std::make_shared<module::MIO>(feature_tensor);
        auto output_mio = std::make_shared<module::MIO>(output_tensor);

        auto feature_node = createFeatureExtractorNode(configs, input_mio, feature_mio);
        if (!feature_node)
        {
            LOG(ERROR, "Failed to create feature extractor node");
            return nullptr;
        }

        auto inference_node = createInferenceNode(configs, feature_mio, output_mio);
        if (!inference_node)
        {
            LOG(ERROR, "Failed to create inference node");
            return nullptr;
        }

        auto *feature_node_ptr = dag->addNode(feature_node);
        auto *inference_node_ptr = dag->addNode(inference_node);

        if (!feature_node_ptr || !inference_node_ptr)
        {
            LOG(ERROR, "Failed to add nodes to DAG");
            return nullptr;
        }

        if (!dag->addEdge(feature_node_ptr, inference_node_ptr))
        {
            LOG(ERROR, "Failed to create edge between nodes");
            return nullptr;
        }

        LOG(INFO, "SoundClassificationDAG created successfully with 2 nodes");
        return dag;
    }

    std::shared_ptr<core::Tensor> SoundClassificationDAGBuilder::createInputTensor()
    {
        LOG(DEBUG, "Creating input tensor: int16[%zu]", AUDIO_SAMPLES);

        core::Tensor::Shape shape(std::vector<int> { static_cast<int>(AUDIO_SAMPLES) });
        return allocateAlignedTensor<int16_t>(shape, core::Tensor::Type::Int16);
    }

    std::shared_ptr<core::Tensor> SoundClassificationDAGBuilder::createFeatureTensor()
    {
        LOG(DEBUG, "Creating feature tensor: float32[%zu]", FEATURE_COUNT);

        core::Tensor::Shape shape(std::vector<int> { static_cast<int>(FEATURE_COUNT) });
        return allocateAlignedTensor<float>(shape, core::Tensor::Type::Float32);
    }

    std::shared_ptr<core::Tensor> SoundClassificationDAGBuilder::createOutputTensor(size_t output_size)
    {
        LOG(DEBUG, "Creating output tensor: float32[%zu]", output_size);

        core::Tensor::Shape shape(std::vector<int> { static_cast<int>(output_size) });
        return allocateAlignedTensor<float>(shape, core::Tensor::Type::Float32);
    }

    std::shared_ptr<module::MNode> SoundClassificationDAGBuilder::createFeatureExtractorNode(
        const core::ConfigMap &configs, std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio)
    {
        LOG(DEBUG, "Creating %s", FEATURE_EXTRACTOR_NODE_NAME);

        const auto &node_builders = module::MNodeBuilderRegistry::getNodeBuilderMap();
        auto it = node_builders.find(FEATURE_EXTRACTOR_NODE_NAME);
        if (it == node_builders.end())
        {
            LOG(ERROR, "%s builder not found in registry", FEATURE_EXTRACTOR_NODE_NAME);
            return nullptr;
        }

        module::MIOS inputs = { input_mio };
        module::MIOS outputs = { output_mio };

        auto node = it->second(configs, &inputs, &outputs, 1);
        if (!node)
        {
            LOG(ERROR, "Failed to create %s", FEATURE_EXTRACTOR_NODE_NAME);
            return nullptr;
        }

        LOG(DEBUG, "%s created successfully", FEATURE_EXTRACTOR_NODE_NAME);
        return node;
    }

    std::shared_ptr<module::MNode> SoundClassificationDAGBuilder::createInferenceNode(const core::ConfigMap &configs,
        std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio)
    {
        LOG(DEBUG, "Creating %s", INFERENCE_NODE_NAME);

        const auto &node_builders = module::MNodeBuilderRegistry::getNodeBuilderMap();
        auto it = node_builders.find(INFERENCE_NODE_NAME);
        if (it == node_builders.end())
        {
            LOG(ERROR, "%s builder not found in registry", INFERENCE_NODE_NAME);
            return nullptr;
        }

        module::MIOS inputs = { input_mio };
        module::MIOS outputs = { output_mio };

        auto node = it->second(configs, &inputs, &outputs, 2);
        if (!node)
        {
            LOG(ERROR, "Failed to create %s", INFERENCE_NODE_NAME);
            return nullptr;
        }

        LOG(DEBUG, "%s created successfully", INFERENCE_NODE_NAME);
        return node;
    }

    template<typename T>
    std::shared_ptr<core::Tensor> SoundClassificationDAGBuilder::allocateAlignedTensor(const core::Tensor::Shape &shape,
        core::Tensor::Type type)
    {
        size_t total_elements = shape.dot();
        size_t total_bytes = total_elements * sizeof(T);

        void *raw_data = heap_caps_aligned_alloc(16, total_bytes, MALLOC_CAP_SPIRAM);
        if (!raw_data)
        {
            LOG(ERROR, "Failed to allocate %zu bytes of aligned memory", total_bytes);
            return nullptr;
        }

        LOG(DEBUG, "Allocated %zu bytes of aligned tensor memory at %p", total_bytes, raw_data);

        std::shared_ptr<std::byte[]> data(reinterpret_cast<std::byte *>(raw_data), [](std::byte *ptr) {
            if (ptr)
            {
                heap_caps_free(ptr);
            }
        });

        auto tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(type, shape, data, total_bytes);

        return tensor;
    }

    size_t SoundClassificationDAGBuilder::getOutputSize(const core::ConfigMap &configs)
    {
        if (auto it = configs.find("output_classes"); it != configs.end())
        {
            if (auto output_classes = std::get_if<int>(&it->second))
            {
                LOG(INFO, "Using explicitly configured output classes: %d", *output_classes);
                return static_cast<size_t>(std::max(1, *output_classes));
            }
        }
        return 100;
    }

    size_t SoundClassificationDAG::getActualOutputSize() const noexcept
    {
        auto output_node = node("SpeechCommandsNode");
        if (output_node && !output_node->outputs().empty())
        {
            auto output_mio = output_node->outputs()[0];
            if (output_mio && output_mio->operator()())
            {
                auto output_tensor = output_mio->operator()();
                return static_cast<size_t>(output_tensor->shape().dot());
            }
        }
        return 0;
    }

    core::Status SoundClassificationDAGBuilder::validateConfig(const core::ConfigMap &configs)
    {
        if (auto it = configs.find("engine_id"); it != configs.end())
        {
            if (auto engine_id = std::get_if<int>(&it->second))
            {
                if (*engine_id < 0)
                {
                    return STATUS(EINVAL, "engine_id must be non-negative");
                }
            }
            else
            {
                return STATUS(EINVAL, "engine_id must be an integer");
            }
        }

        if (auto it = configs.find("model_id"); it != configs.end())
        {
            if (auto model_id = std::get_if<int>(&it->second))
            {
                if (*model_id < 0)
                {
                    return STATUS(EINVAL, "model_id must be non-negative");
                }
            }
            else
            {
                return STATUS(EINVAL, "model_id must be an integer");
            }
        }

        if (auto it = configs.find("graph_id"); it != configs.end())
        {
            if (auto graph_id = std::get_if<int>(&it->second))
            {
                if (*graph_id < 0)
                {
                    return STATUS(EINVAL, "graph_id must be non-negative");
                }
            }
            else
            {
                return STATUS(EINVAL, "graph_id must be an integer");
            }
        }

        if (auto it = configs.find("output_classes"); it != configs.end())
        {
            if (auto output_classes = std::get_if<int>(&it->second))
            {
                if (*output_classes < 1 || *output_classes > 1000)
                {
                    return STATUS(EINVAL, "output_classes must be between 1 and 1000");
                }
            }
            else
            {
                return STATUS(EINVAL, "output_classes must be an integer");
            }
        }

        return STATUS_OK();
    }

    template std::shared_ptr<core::Tensor> SoundClassificationDAGBuilder::allocateAlignedTensor<int16_t>(
        const core::Tensor::Shape &shape, core::Tensor::Type type);
    template std::shared_ptr<core::Tensor> SoundClassificationDAGBuilder::allocateAlignedTensor<float>(
        const core::Tensor::Shape &shape, core::Tensor::Type type);

    std::shared_ptr<module::MDAG> createSoundClassificationDAG(const core::ConfigMap &configs)
    {
        LOG(DEBUG, "Creating SoundClassificationDAG via builder function");

        return SoundClassificationDAGBuilder::create(configs);
    }

}} // namespace algorithms::dag
