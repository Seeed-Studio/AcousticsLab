#include "feature_extractor_node.hpp"
#include "core/logger.hpp"

namespace porting { namespace algorithms { namespace node {

    core::Status FeatureExtractorNode::validateTensors(const module::MIOS &inputs,
        const module::MIOS &outputs) const noexcept
    {
        if (inputs.size() != 1)
        {
            return STATUS(EINVAL, "FeatureExtractorNode requires exactly 1 input tensor");
        }

        if (outputs.size() != 1)
        {
            return STATUS(EINVAL, "FeatureExtractorNode requires exactly 1 output tensor");
        }

        auto input_tensor = inputs[0]->operator()();
        auto output_tensor = outputs[0]->operator()();

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
            LOG(ERROR, "Output tensor data type mismatch: expected %d, got %d", static_cast<int>(getOutputDataType()),
                static_cast<int>(output_tensor->dtype()));
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

}}} // namespace porting::algorithms::node
