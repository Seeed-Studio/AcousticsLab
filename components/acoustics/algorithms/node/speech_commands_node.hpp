#pragma once
#ifndef SPEECH_COMMANDS_NODE_HPP
#define SPEECH_COMMANDS_NODE_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/model.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "hal/engine.hpp"
#include "module/module_node.hpp"

#include <memory>
#include <string>

namespace algorithms { namespace node {

    class SpeechCommands final: public module::MNode
    {
    public:
        inline SpeechCommands(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs, int priority)
            : module::MNode("SpeechCommands", std::move(inputs), std::move(outputs), priority), _engine(nullptr),
              _model(nullptr), _graph(nullptr), _engine_id(1), _model_id(1), _graph_id(0), _initialized(false),
              _model_output_classes(0)
        {
            LOG(DEBUG, "Creating SpeechCommands with priority %d", priority);
            LOG(INFO, "SpeechCommands configured: engine_id=%d, model_id=%d, graph_id=%d", _engine_id, _model_id,
                _graph_id);
        }

        inline ~SpeechCommands() override
        {
            LOG(DEBUG, "Destroying SpeechCommands");
        }

        inline core::Status config(const core::ConfigMap &configs) noexcept override
        {
            LOG(DEBUG, "Reconfiguring SpeechCommands");
            LOG(INFO, "SpeechCommands using default configuration: engine_id=%d, model_id=%d, graph_id=%d", _engine_id,
                _model_id, _graph_id);
            return STATUS_OK();
        }

        inline core::Status initialize() noexcept
        {
            LOG(DEBUG, "Initializing SpeechCommands");

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

            auto model_info
                = _engine->modelInfo([this](const core::Model::Info &info) { return info.id == _model_id; });
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
            if (!model_output_tensor)
            {
                return STATUS(EFAULT, "Model graph has no output tensor");
            }

            _model_output_classes = static_cast<size_t>(model_output_tensor->shape().dot());
            LOG(INFO, "Model initialized with %zu output classes", _model_output_classes);

            _initialized = true;
            LOG(INFO, "SpeechCommands initialized successfully");

            return STATUS_OK();
        }

        inline core::Status validateTensors(const module::MIOS &inputs, const module::MIOS &outputs) const noexcept
        {
            if (inputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommands requires exactly 1 input tensor");
            }

            if (outputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommands requires exactly 1 output tensor");
            }

            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

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

        inline size_t getModelOutputClasses() const noexcept
        {
            return _model_output_classes;
        }

    protected:
        inline core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override
        {
            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

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

            auto status = copyInputData(input_tensor, model_input_tensor);
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

            if (output_tensor->dtype() != core::Tensor::Type::Class)
            {
                return STATUS(EINVAL, "Output tensor must be of class_t type");
            }

            const core::Tensor::Shape new_shape(1, static_cast<int>(_model_output_classes));
            if (!output_tensor->reshape(new_shape))
            {
                return STATUS(EINVAL, "Failed to reshape output tensor");
            }

            core::class_t *class_data = output_tensor->data<core::class_t>();
            if (!class_data)
            {
                return STATUS(EFAULT, "Failed to get class_t data pointer");
            }

            if (model_output_tensor->dtype() == core::Tensor::Type::Float32)
            {
                const float *model_data = model_output_tensor->data<float>();
                if (!model_data)
                {
                    return STATUS(EFAULT, "Failed to get model output data pointer");
                }

                for (size_t i = 0; i < _model_output_classes; ++i)
                {
                    class_data[i] = { static_cast<int>(i), model_data[i] };
                }
                LOG(DEBUG, "Processed %zu float32 output elements to class_t", _model_output_classes);
            }
            else if (model_output_tensor->dtype() == core::Tensor::Type::Int8)
            {
                const int8_t *model_data = model_output_tensor->data<int8_t>();
                if (!model_data)
                {
                    return STATUS(EFAULT, "Failed to get model output data pointer");
                }

                auto quant_params = _graph->outputQuantParams(0);
                float scale = quant_params.scale();
                int32_t zero_point = quant_params.zeroPoint();

                for (size_t i = 0; i < _model_output_classes; ++i)
                {
                    float confidence = static_cast<float>(model_data[i] - zero_point) * scale;
                    class_data[i] = { static_cast<int>(i), confidence };
                }
                LOG(DEBUG, "Dequantized and processed %zu int8 output elements to class_t (scale=%.6f, zero_point=%ld)",
                    _model_output_classes, scale, (long)zero_point);
            }
            else
            {
                return STATUS(ENOTSUP, "Unsupported model output tensor data type");
            }

            return STATUS_OK();
        }

    private:
        std::shared_ptr<hal::Engine> _engine;
        std::shared_ptr<core::Model> _model;
        std::shared_ptr<core::Model::Graph> _graph;

        int _engine_id;
        int _model_id;
        int _graph_id;

        bool _initialized;
        size_t _model_output_classes;

        inline core::Status copyInputData(core::Tensor *input_tensor, core::Tensor *model_input_tensor) const noexcept
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
                        quantized_value = std::max(static_cast<int32_t>(-128),
                            std::min(static_cast<int32_t>(127), quantized_value));
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
    };

    inline std::shared_ptr<module::MNode> createSpeechCommands(const core::ConfigMap &configs, module::MIOS *inputs,
        module::MIOS *outputs, int priority)
    {
        LOG(DEBUG, "Creating SpeechCommands via builder function");

        module::MIOS input_mios = inputs ? *inputs : module::MIOS {};
        module::MIOS output_mios = outputs ? *outputs : module::MIOS {};

        auto node = std::make_shared<SpeechCommands>(configs, std::move(input_mios), std::move(output_mios), priority);

        auto init_status = node->initialize();
        if (!init_status)
        {
            LOG(ERROR, "Failed to initialize SpeechCommands: %s", init_status.message().c_str());
            return nullptr;
        }

        module::MIOS validate_inputs = inputs ? *inputs : module::MIOS {};
        module::MIOS validate_outputs = outputs ? *outputs : module::MIOS {};
        auto validate_status = node->validateTensors(validate_inputs, validate_outputs);
        if (!validate_status)
        {
            LOG(ERROR, "SpeechCommands tensor validation failed: %s", validate_status.message().c_str());
            return nullptr;
        }

        return node;
    }

}} // namespace algorithms::node

#endif // SPEECH_COMMANDS_NODE_HPP
