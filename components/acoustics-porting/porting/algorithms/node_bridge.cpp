#include "core/logger.hpp"
#include "module/module_node.hpp"
#include "node/esp_feature_extractor_node.hpp"

namespace bridge {

void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__()
{
    LOG(INFO, "Registering external module node builders");

    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("ESPFeatureExtractorNode",
        porting::algorithms::node::createESPFeatureExtractorNode, false);

    if (!status)
    {
        LOG(ERROR, "Failed to register ESPFeatureExtractorNode: %s", status.message().c_str());
    }
    else
    {
        LOG(INFO, "Successfully registered ESPFeatureExtractorNode");
    }

    const auto &node_map = module::MNodeBuilderRegistry::getNodeBuilderMap();
    LOG(INFO, "Total registered nodes (internal + external): %zu", node_map.size());
    for (const auto &[name, builder]: node_map)
    {
        LOG(DEBUG, "  - %s", name.data());
    }
}

} // namespace bridge
