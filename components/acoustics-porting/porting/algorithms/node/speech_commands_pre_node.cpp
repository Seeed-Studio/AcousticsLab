#include "speech_commands_pre_node.hpp"
#include "core/logger.hpp"
#include "module/module_node.hpp"

namespace bridge {

void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__()
{
    LOG(INFO, "Registering external module node builders");

    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("SpeechCommandsPreprocess",
        porting::algorithms::node::createSpeechCommandsPreprocess, false);

    if (!status)
    {
        LOG(ERROR, "Failed to register SpeechCommandsPreprocess: %s", status.message().c_str());
    }
    else
    {
        LOG(INFO, "Successfully registered SpeechCommandsPreprocess");
    }

    const auto &node_map = module::MNodeBuilderRegistry::getNodeBuilderMap();
    LOG(INFO, "Total registered nodes (internal + external): %zu", node_map.size());
#if LOG_LEVEL >= DEBUG
    for (const auto &[name, builder]: node_map)
    {
        LOG(DEBUG, "  - %s", name.data());
    }
#endif
}

} // namespace bridge
