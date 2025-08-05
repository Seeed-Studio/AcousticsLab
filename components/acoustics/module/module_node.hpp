#pragma once
#ifndef MODULE_NODE_HPP
#define MODULE_NODE_HPP

#include "module_io.hpp"

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/reporter.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace module {

class MNode;

class MNodeBuilderRegistry final
{
public:
    using NodeBuilder = std::shared_ptr<MNode> (*)(const core::ConfigMap &, MIOS *, MIOS *, int);
    using NodeBuilderMap = std::unordered_map<std::string_view, NodeBuilder>;

    MNodeBuilderRegistry() = default;
    ~MNodeBuilderRegistry() = default;

    inline static std::shared_ptr<MNode> getNode(std::string_view name, const core::ConfigMap &configs = {},
        MIOS *inputs = nullptr, MIOS *outputs = nullptr, int priority = 0) noexcept
    {
        auto it = _nodes.find(name);
        if (it != _nodes.end()) [[likely]]
        {
            return it->second(configs, inputs, outputs, priority);
        }
        return {};
    }

    static const NodeBuilderMap &getNodeBuilderMap() noexcept
    {
        return _nodes;
    }

    static core::Status registerNodeBuilder(std::string_view name, NodeBuilder builder, bool replace = false) noexcept;

private:
    static NodeBuilderMap _nodes;
};

class MNode
{
public:
    struct Lower final
    {
        inline bool operator()(const MNode *lhs, const MNode *rhs) const noexcept
        {
            const auto lhs_priority = lhs ? lhs->priority() : 0;
            const auto rhs_priority = rhs ? rhs->priority() : 0;
            return lhs_priority < rhs_priority;
        }
    };

    virtual ~MNode() = default;

    const std::string &name() const noexcept
    {
        return _name;
    }

    int priority() const noexcept
    {
        return _priority;
    }

    inline const MIOS &inputs() const noexcept
    {
        return _inputs;
    }

    inline const MIOS &outputs() const noexcept
    {
        return _outputs;
    }

    std::shared_ptr<MIO> input(size_t index) const noexcept
    {
        if (index < _inputs.size()) [[likely]]
        {
            return _inputs[index];
        }
        return {};
    }

    std::shared_ptr<MIO> input(std::string_view attribute) const noexcept
    {
        for (const auto &input: _inputs)
        {
            if (input->attribute() == attribute)
            {
                return input;
            }
        }
        return {};
    }

    std::shared_ptr<MIO> output(size_t index) const noexcept
    {
        if (index < _outputs.size()) [[likely]]
        {
            return _outputs[index];
        }
        return {};
    }

    std::shared_ptr<MIO> output(std::string_view attribute) const noexcept
    {
        for (const auto &output: _outputs)
        {
            if (output->attribute() == attribute)
            {
                return output;
            }
        }
        return {};
    }

    virtual core::Status config(const core::ConfigMap &configs) noexcept
    {
        return STATUS(ENOTSUP, "Configuration not supported for this node");
    }

    inline core::Status operator()() noexcept
    {
        return forward(_inputs, _outputs);
    }

    inline core::Status operator()(core::Reporter &reporter) noexcept
    {
        const auto start = std::chrono::steady_clock::now();
        auto status = forward(_inputs, _outputs);
        const auto end = std::chrono::steady_clock::now();
        reporter.time_micro.insert_or_assign(_name, std::chrono::duration_cast<std::chrono::microseconds>(end - start));
        return status;
    }

protected:
    template<typename IS, typename OS>
    constexpr explicit MNode(std::string name, IS &&inputs, OS &&outputs, int priority) noexcept
        : _name(std::move(name)), _inputs(std::forward<IS>(inputs)), _outputs(std::forward<OS>(outputs)),
          _priority(priority)
    {
        LOG(DEBUG, "Module node '%s' created with priority %d", _name.c_str(), _priority);
    }

    virtual inline core::Status forward(const MIOS &inputs, MIOS &outputs) noexcept = 0;

private:
    const std::string _name;
    MIOS _inputs;
    MIOS _outputs;
    const int _priority;
};

} // namespace module

namespace bridge {

extern void __REGISTER_INTERNAL_MODULE_NODE_BUILDER__();
extern void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__();

} // namespace bridge

#endif
