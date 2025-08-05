#pragma once
#ifndef FEATURE_EXTRACTOR_NODE_HPP
#define FEATURE_EXTRACTOR_NODE_HPP

#include "core/config_object.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_node.hpp"

#include <memory>

namespace porting { namespace algorithms { namespace node {

    class FeatureExtractorNode: public module::MNode
    {
    public:
        template<typename IS, typename OS>
        constexpr explicit FeatureExtractorNode(std::string name, IS &&inputs, OS &&outputs, int priority) noexcept
            : module::MNode(std::move(name), std::forward<IS>(inputs), std::forward<OS>(outputs), priority)
        {
        }

        virtual ~FeatureExtractorNode() = default;

        virtual size_t getInputSampleCount() const noexcept = 0;

        virtual size_t getOutputFeatureCount() const noexcept = 0;

        virtual core::Tensor::Type getInputDataType() const noexcept = 0;

        virtual core::Tensor::Type getOutputDataType() const noexcept = 0;

    protected:
        virtual core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept = 0;

        virtual core::Status validateTensors(const module::MIOS &inputs, const module::MIOS &outputs) const noexcept;
    };

}}} // namespace porting::algorithms::node

#endif // FEATURE_EXTRACTOR_NODE_HPP
