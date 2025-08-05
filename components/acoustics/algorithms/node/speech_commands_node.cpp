#include "speech_commands_node.hpp"
#include "core/logger.hpp"
#include "hal/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace algorithms { namespace node {

    SpeechCommandsNode::SpeechCommandsNode(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs,
        int priority)
        : module::MNode("SpeechCommandsNode", std::move(inputs), std::move(outputs), priority), _engine(nullptr),
          _model(nullptr), _graph(nullptr), _engine_id(1), _model_id(1), _graph_id(0), _initialized(false)
    {
        LOG(DEBUG, "Creating SpeechCommandsNode with priority %d", priority);

        if (auto it = configs.find("engine_id"); it != configs.end())
        {
            if (auto engine_id = std::get_if<int>(&it->second))
            {
                _engine_id = *engine_id;
            }
        }

        if (auto it = configs.find("model_id"); it != configs.end())
        {
            if (auto model_id = std::get_if<int>(&it->second))
            {
                _model_id = *model_id;
            }
        }

        if (auto it = configs.find("graph_id"); it != configs.end())
        {
            if (auto graph_id = std::get_if<int>(&it->second))
            {
                _graph_id = *graph_id;
            }
        }

        LOG(INFO, "SpeechCommandsNode configured: engine_id=%d, model_id=%d, graph_id=%d", _engine_id, _model_id,
            _graph_id);
    }

    SpeechCommandsNode::~SpeechCommandsNode()
    {
        LOG(DEBUG, "Destroying SpeechCommandsNode");
    }

    core::Status SpeechCommandsNode::config(const core::ConfigMap &configs) noexcept
    {
        LOG(DEBUG, "Reconfiguring SpeechCommandsNode");

        bool config_changed = false;

        if (auto it = configs.find("engine_id"); it != configs.end())
        {
            if (auto engine_id = std::get_if<int>(&it->second))
            {
                if (*engine_id != _engine_id)
                {
                    _engine_id = *engine_id;
                    config_changed = true;
                }
            }
        }

        if (auto it = configs.find("model_id"); it != configs.end())
        {
            if (auto model_id = std::get_if<int>(&it->second))
            {
                if (*model_id != _model_id)
                {
                    _model_id = *model_id;
                    config_changed = true;
                }
            }
        }

        if (auto it = configs.find("graph_id"); it != configs.end())
        {
            if (auto graph_id = std::get_if<int>(&it->second))
            {
                if (*graph_id != _graph_id)
                {
                    _graph_id = *graph_id;
                    config_changed = true;
                }
            }
        }

        if (config_changed)
        {
            _initialized = false;
            _engine.reset();
            _model.reset();
            _graph.reset();

            LOG(INFO, "SpeechCommandsNode reconfigured: engine_id=%d, model_id=%d, graph_id=%d", _engine_id, _model_id,
                _graph_id);
        }

        return STATUS_OK();
    }

    core::Status SpeechCommandsNode::forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept
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

        if (!input_tensor || !output_tensor)
        {
            return STATUS(EINVAL, "Input or output tensor is null");
        }

        auto model_input_tensor = _graph->input(0);
        auto model_output_tensor = _graph->output(0);

        if (!model_input_tensor || !model_output_tensor)
        {
            return STATUS(EFAULT, "Model input or output tensor is null");
        }

        status = copyInputData(input_tensor, model_input_tensor);
        if (!status)
        {
            return status;
        }

        status = _graph->forward();
        if (!status)
        {
            LOG(ERROR, "Model inference failed: %s", status.message().c_str());
            return status;
        }

        status = copyOutputData(model_output_tensor, output_tensor);
        if (!status)
        {
            return status;
        }

        LOG(DEBUG, "InferenceNode forward completed successfully");

        return STATUS_OK();
    }

    core::Status SpeechCommandsNode::initialize() noexcept
    {
        LOG(DEBUG, "Initializing InferenceNode");

        _engine = std::shared_ptr<hal::Engine>(hal::EngineRegistry::getEngine(_engine_id), [](hal::Engine *) { });
        if (!_engine)
        {
            return STATUS(ENODEV, "Engine not found");
        }

        if (!_engine->initialized())
        {
            auto status = _engine->init();
            if (!status)
            {
                return STATUS(EFAULT, "Failed to initialize engine");
            }
        }

        auto model_info = _engine->modelInfo([this](const core::Model::Info &info) { return info.id == _model_id; });
        if (!model_info)
        {
            return STATUS(ENOENT, "Model not found");
        }

        auto status = _engine->loadModel(model_info, _model);
        if (!status)
        {
            return STATUS(EFAULT, "Failed to load model");
        }

        _graph = _model->graph(_graph_id);
        if (!_graph)
        {
            return STATUS(ENOENT, "Graph not found");
        }

        auto model_output_tensor = _graph->output(0);
        if (model_output_tensor)
        {
            size_t actual_output_size = static_cast<size_t>(model_output_tensor->shape().dot());
            LOG(INFO, "Model actual output size: %zu classes", actual_output_size);
        }
        else
        {
            LOG(WARNING, "Could not get model output tensor information");
        }

        _initialized = true;
        LOG(INFO, "SpeechCommandsNode initialized successfully");

        return STATUS_OK();
    }

    core::Status SpeechCommandsNode::validateTensors(const module::MIOS &inputs,
        const module::MIOS &outputs) const noexcept
    {
        if (inputs.size() != 1)
        {
            return STATUS(EINVAL, "InferenceNode requires exactly 1 input tensor");
        }

        if (outputs.size() != 1)
        {
            return STATUS(EINVAL, "InferenceNode requires exactly 1 output tensor");
        }

        auto input_tensor = inputs[0]->operator()();
        auto output_tensor = outputs[0]->operator()();

        if (!input_tensor || !output_tensor)
        {
            return STATUS(EINVAL, "Input or output tensor is null");
        }

        auto model_input_tensor = _graph->input(0);
        if (input_tensor->shape().dot() != model_input_tensor->shape().dot())
        {
            LOG(ERROR, "Input tensor size mismatch: expected %d, got %d", model_input_tensor->shape().dot(),
                input_tensor->shape().dot());
            return STATUS(EINVAL, "Input tensor size mismatch");
        }

        auto model_output_tensor = _graph->output(0);
        size_t model_output_size = static_cast<size_t>(model_output_tensor->shape().dot());
        size_t dag_output_size = static_cast<size_t>(output_tensor->shape().dot());

        if (dag_output_size < model_output_size)
        {
            LOG(ERROR, "Output tensor too small: model needs %zu, DAG provides %zu", model_output_size,
                dag_output_size);
            return STATUS(EINVAL, "Output tensor too small for model output");
        }
        else if (dag_output_size > model_output_size)
        {
            LOG(INFO, "Output tensor larger than model output: model=%zu, DAG=%zu (will use first %zu elements)",
                model_output_size, dag_output_size, model_output_size);
        }
        else
        {
            LOG(INFO, "Output tensor size matches model output: %zu elements", model_output_size);
        }

        return STATUS_OK();
    }

    core::Status SpeechCommandsNode::copyInputData(core::Tensor *input_tensor,
        core::Tensor *model_input_tensor) const noexcept
    {
        const size_t elements_to_copy = std::min(static_cast<size_t>(input_tensor->shape().dot()),
            static_cast<size_t>(model_input_tensor->shape().dot()));

        if (elements_to_copy == 0)
        {
            return STATUS(EINVAL, "No elements to copy");
        }

        if (model_input_tensor->dtype() == core::Tensor::Type::Float32)
        {
            float *model_data = model_input_tensor->data<float>();
            if (!model_data)
            {
                return STATUS(EFAULT, "Failed to get model input data pointer");
            }

            if (input_tensor->dtype() == core::Tensor::Type::Float32)
            {
                const float *input_data = input_tensor->data<float>();
                if (!input_data)
                {
                    return STATUS(EFAULT, "Failed to get input data pointer");
                }
                std::memcpy(model_data, input_data, elements_to_copy * sizeof(float));
                LOG(DEBUG, "Copied %zu float32 elements (direct copy)", elements_to_copy);
            }
            else if (input_tensor->dtype() == core::Tensor::Type::Int16)
            {
                const int16_t *input_data = input_tensor->data<int16_t>();
                if (!input_data)
                {
                    return STATUS(EFAULT, "Failed to get input data pointer");
                }
                for (size_t i = 0; i < elements_to_copy; ++i)
                {
                    model_data[i] = static_cast<float>(input_data[i]) / 32768.0f;
                }
                LOG(DEBUG, "Converted and copied %zu int16 to float32 elements", elements_to_copy);
            }
            else
            {
                return STATUS(ENOTSUP, "Unsupported input tensor data type for float32 model");
            }
        }
        else if (model_input_tensor->dtype() == core::Tensor::Type::Int8)
        {
            int8_t *model_data = model_input_tensor->data<int8_t>();
            if (!model_data)
            {
                return STATUS(EFAULT, "Failed to get model input data pointer");
            }

            auto input_quant_params = _graph->inputQuantParams(0);
            float scale = input_quant_params.scale();
            int32_t zero_point = input_quant_params.zeroPoint();

            if (input_tensor->dtype() == core::Tensor::Type::Float32)
            {
                const float *input_data = input_tensor->data<float>();
                if (!input_data)
                {
                    return STATUS(EFAULT, "Failed to get input data pointer");
                }

                for (size_t i = 0; i < elements_to_copy; ++i)
                {
                    int32_t quantized_value = static_cast<int32_t>(std::round(input_data[i] / scale)) + zero_point;
                    quantized_value
                        = std::max(static_cast<int32_t>(-128), std::min(static_cast<int32_t>(127), quantized_value));
                    model_data[i] = static_cast<int8_t>(quantized_value);
                }
                LOG(DEBUG, "Quantized and copied %zu float32 to int8 elements (scale=%.6f, zero_point=%ld)",
                    elements_to_copy, scale, (long)zero_point);
            }
            else
            {
                return STATUS(ENOTSUP, "Unsupported input tensor data type for int8 model");
            }
        }
        else
        {
            return STATUS(ENOTSUP, "Unsupported model input tensor data type");
        }

        return STATUS_OK();
    }

    core::Status SpeechCommandsNode::copyOutputData(core::Tensor *model_output_tensor,
        core::Tensor *output_tensor) const noexcept
    {
        const size_t model_output_size = static_cast<size_t>(model_output_tensor->shape().dot());
        const size_t output_tensor_capacity = static_cast<size_t>(output_tensor->shape().dot());
        const size_t elements_to_copy = std::min(model_output_size, output_tensor_capacity);

        if (elements_to_copy == 0)
        {
            return STATUS(EINVAL, "No elements to copy");
        }

        LOG(DEBUG, "Model output size: %zu, Output tensor capacity: %zu, Copying: %zu elements", model_output_size,
            output_tensor_capacity, elements_to_copy);

        if (model_output_tensor->dtype() == core::Tensor::Type::Float32)
        {
            const float *model_data = model_output_tensor->data<float>();
            if (!model_data)
            {
                return STATUS(EFAULT, "Failed to get model output data pointer");
            }

            if (output_tensor->dtype() == core::Tensor::Type::Float32)
            {
                float *output_data = output_tensor->data<float>();
                if (!output_data)
                {
                    return STATUS(EFAULT, "Failed to get output data pointer");
                }
                std::memcpy(output_data, model_data, elements_to_copy * sizeof(float));
                LOG(DEBUG, "Copied %zu float32 output elements (direct copy)", elements_to_copy);
            }
            else
            {
                return STATUS(ENOTSUP, "Unsupported output tensor data type for float32 model");
            }
        }
        else if (model_output_tensor->dtype() == core::Tensor::Type::Int8)
        {
            const int8_t *model_data = model_output_tensor->data<int8_t>();
            if (!model_data)
            {
                return STATUS(EFAULT, "Failed to get model output data pointer");
            }

            auto output_quant_params = _graph->outputQuantParams(0);
            float scale = output_quant_params.scale();
            int32_t zero_point = output_quant_params.zeroPoint();

            if (output_tensor->dtype() == core::Tensor::Type::Float32)
            {
                float *output_data = output_tensor->data<float>();
                if (!output_data)
                {
                    return STATUS(EFAULT, "Failed to get output data pointer");
                }

                for (size_t i = 0; i < elements_to_copy; ++i)
                {
                    output_data[i] = static_cast<float>(model_data[i] - zero_point) * scale;
                }
                LOG(DEBUG, "Dequantized and copied %zu int8 to float32 output elements (scale=%.6f, zero_point=%ld)",
                    elements_to_copy, scale, (long)zero_point);
            }
            else
            {
                return STATUS(ENOTSUP, "Unsupported output tensor data type for int8 model");
            }
        }
        else
        {
            return STATUS(ENOTSUP, "Unsupported model output tensor data type");
        }

        if (output_tensor_capacity > model_output_size && output_tensor->dtype() == core::Tensor::Type::Float32)
        {
            float *output_data = output_tensor->data<float>();
            if (output_data)
            {
                // Zero out the unused portion
                std::memset(output_data + model_output_size, 0,
                    (output_tensor_capacity - model_output_size) * sizeof(float));
                LOG(DEBUG, "Cleared unused portion of output tensor: elements %zu-%zu", model_output_size,
                    output_tensor_capacity - 1);
            }
        }

        return STATUS_OK();
    }

    size_t SpeechCommandsNode::getModelOutputSize() const noexcept
    {
        if (!_initialized || !_graph)
        {
            return 0;
        }

        auto model_output_tensor = _graph->output(0);
        if (!model_output_tensor)
        {
            return 0;
        }

        return static_cast<size_t>(model_output_tensor->shape().dot());
    }

    std::shared_ptr<module::MNode> createSpeechCommandsNode(const core::ConfigMap &configs, module::MIOS *inputs,
        module::MIOS *outputs, int priority)
    {
        LOG(DEBUG, "Creating SpeechCommandsNode via builder function");

        module::MIOS input_mios = inputs ? *inputs : module::MIOS {};
        module::MIOS output_mios = outputs ? *outputs : module::MIOS {};

        return std::make_shared<SpeechCommandsNode>(configs, std::move(input_mios), std::move(output_mios), priority);
    }

}} // namespace algorithms::node
