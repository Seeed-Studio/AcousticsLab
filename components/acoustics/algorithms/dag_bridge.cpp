#include "core/logger.hpp"
#include "dag/sound_classification_dag.hpp"
#include "module/module_dag.hpp"

namespace bridge {

void __REGISTER_MODULE_DAG_BUILDER__()
{
    LOG(INFO, "Registering module DAG builders");

    // Register SoundClassificationDAG
    auto status = module::MDAGBuilderRegistry::registerDAGBuilder("SoundClassificationDAG",
        algorithms::dag::createSoundClassificationDAG);

    if (!status)
    {
        LOG(ERROR, "Failed to register SoundClassificationDAG: %s", status.message().c_str());
    }
    else
    {
        LOG(INFO, "Successfully registered SoundClassificationDAG");
    }

    // Log all registered DAGs for debugging
    const auto &dag_map = module::MDAGBuilderRegistry::getDAGBuilderMap();
    LOG(INFO, "Total registered DAGs: %zu", dag_map.size());
    for (const auto &[name, builder]: dag_map)
    {
        LOG(DEBUG, "  - %s", name.data());
    }
}

} // namespace bridge
