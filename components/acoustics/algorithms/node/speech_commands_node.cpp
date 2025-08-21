#include "speech_commands_node.hpp"
#include "core/logger.hpp"
#include "module/module_node.hpp"

namespace bridge {

void __REGISTER_INTERNAL_MODULE_NODE_BUILDER__()
{
    LOG(INFO, "Registering internal module node builders");

    // Register SpeechCommandsNode
    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("SpeechCommands",
        algorithms::node::createSpeechCommands, false);

    if (!status)
    {
        LOG(ERROR, "Failed to register SpeechCommands: %s", status.message().c_str());
    }
    else
    {
        LOG(INFO, "Successfully registered SpeechCommands");
    }

    // Log all registered nodes for debugging
    const auto &node_map = module::MNodeBuilderRegistry::getNodeBuilderMap();
    LOG(INFO, "Total registered internal nodes: %zu", node_map.size());
#if LOG_LEVEL >= DEBUG
    for (const auto &[name, builder]: node_map)
    {
        LOG(DEBUG, "  - %s", name.data());
    }
#endif
}

} // namespace bridge
