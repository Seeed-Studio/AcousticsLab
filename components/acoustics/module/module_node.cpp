#include "module_node.hpp"

namespace module {

core::Status MNodeBuilderRegistry::registerNodeBuilder(std::string_view name, NodeBuilder builder,
    bool replace) noexcept
{
    if (name.empty() || !builder) [[unlikely]]
    {
        LOG(ERROR, "Invalid node name or builder function");
        return STATUS(EINVAL, "Node name or builder function is invalid");
    }

    auto it = _nodes.find(name);
    if (it != _nodes.end()) [[unlikely]]
    {
        if (!replace)
        {
            return STATUS(EEXIST, "Node with this name already exists");
        }
        it->second = std::move(builder);
    }
    else
    {
        _nodes.emplace(name, std::move(builder));
    }

    return STATUS_OK();
}

MNodeBuilderRegistry::NodeBuilderMap MNodeBuilderRegistry::_nodes = {};

} // namespace module
