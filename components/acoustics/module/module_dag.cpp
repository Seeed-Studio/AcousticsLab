#include "module_dag.hpp"

namespace module {

core::Status MDAGBuilderRegistry::registerDAGBuilder(std::string_view name, DAGBuilder builder) noexcept
{
    if (name.empty()) [[unlikely]]
    {
        LOG(ERROR, "DAG name cannot be empty");
        return STATUS(EINVAL, "DAG name cannot be empty");
    }
    if (!builder) [[unlikely]]
    {
        LOG(ERROR, "DAG builder cannot be null");
        return STATUS(EINVAL, "DAG builder cannot be null");
    }

    auto it = _dags.find(name);
    if (it != _dags.end()) [[unlikely]]
    {
        LOG(WARNING, "DAG builder for %s already exists", name.data());
        return STATUS(EEXIST, "DAG builder already exists");
    }

    _dags.emplace(name, std::move(builder));

    return STATUS_OK();
}

MDAGBuilderRegistry::DAGBuilderMap MDAGBuilderRegistry::_dags;

} // namespace module
