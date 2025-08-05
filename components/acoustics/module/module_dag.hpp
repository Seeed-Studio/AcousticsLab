#pragma once
#ifndef MODULE_DAG_HPP
#define MODULE_DAG_HPP

#include "module_node.hpp"

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <forward_list>
#include <functional>
#include <memory>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace module {

class MDAG;

class MDAGBuilderRegistry final
{
public:
    using DAGBuilder = std::shared_ptr<MDAG> (*)(const core::ConfigMap &);
    using DAGBuilderMap = std::unordered_map<std::string_view, DAGBuilder>;

    MDAGBuilderRegistry() = default;
    ~MDAGBuilderRegistry() = default;

    inline static std::shared_ptr<MDAG> getDAG(std::string_view name, const core::ConfigMap &configs) noexcept
    {
        auto it = _dags.find(name);
        if (it != _dags.end()) [[likely]]
        {
            return it->second(configs);
        }
        return {};
    }

    static const DAGBuilderMap &getDAGBuilderMap() noexcept
    {
        return _dags;
    }

    static core::Status registerDAGBuilder(std::string_view name, DAGBuilder builder) noexcept;

private:
    static DAGBuilderMap _dags;
};

class MDAG
{
public:
    explicit MDAG(std::string_view name) noexcept : _name(name), _nodes(), _adj(), _in_degree(), _execution_order()
    {
        if (name.empty()) [[unlikely]]
        {
            LOG(WARNING, "DAG name cannot be empty");
        }
    }

    ~MDAG() = default;

    MNode *addNode(std::shared_ptr<MNode> node) noexcept
    {
        if (!node) [[unlikely]]
        {
            LOG(ERROR, "Attempted to add a null node to the DAG");
            return nullptr;
        }

        auto ptr = node.get();
        if (std::find_if(_nodes.begin(), _nodes.end(),
                [ptr](const std::shared_ptr<MNode> &n) { return n.get() == ptr; })
            != _nodes.end()) [[unlikely]]
        {
            LOG(WARNING, "Node %s already exists in the DAG", node->name().data());
            return ptr;
        }

        _execution_order.clear();

        _nodes.push_front(std::move(node));
        _adj.try_emplace(ptr, std::forward_list<MNode *> {});
        _in_degree.try_emplace(ptr, 0);

        return ptr;
    }

    bool addEdge(MNode *from, MNode *to) noexcept
    {
        if (!from || !to) [[unlikely]]
        {
            LOG(ERROR, "Attempted to add an edge with null nodes");
            return false;
        }
        if (_adj.find(from) == _adj.end() || _adj.find(to) == _adj.end()) [[unlikely]]
        {
            LOG(ERROR, "One or both nodes not found in the DAG");
            return false;
        }

        _execution_order.clear();

        _adj[from].push_front(to);
        ++_in_degree[to];

        return true;
    }

    bool computeExecutionOrder() const noexcept
    {
        std::priority_queue<MNode *, std::vector<MNode *>, MNode::Lower> pq;
        auto in_degree = _in_degree;

        for (const auto &id: in_degree)
        {
            if (id.second == 0) [[likely]]
            {
                pq.push(id.first);
            }
        }

        while (!pq.empty())
        {
            auto *node = pq.top();
            pq.pop();
            _execution_order.push_back(node);

            for (auto *neighbor: _adj.at(node))
            {
                if (--in_degree[neighbor] == 0) [[likely]]
                {
                    pq.push(neighbor);
                }
            }
        }

        if (_execution_order.size() != std::distance(_nodes.begin(), _nodes.end())) [[unlikely]]
        {
            LOG(ERROR, "Cycle detected in the DAG, execution order incomplete");
            _execution_order.clear();
            return false;
        }

        _execution_order.shrink_to_fit();

        return true;
    }

    MNode *node(std::string_view name) const noexcept
    {
        for (const auto &node: _nodes)
        {
            if (node->name() == name)
            {
                return node.get();
            }
        }
        return nullptr;
    }

    const std::forward_list<std::shared_ptr<module::MNode>> &nodes() const noexcept
    {
        return _nodes;
    }

    virtual std::shared_ptr<core::Tensor> getInputTensor(size_t index = 0) const noexcept
    {
        return nullptr;
    }

    virtual std::shared_ptr<core::Tensor> getOutputTensor(size_t index = 0) const noexcept
    {
        return nullptr;
    }

    inline core::Status operator()() noexcept
    {
        if (_execution_order.empty()) [[unlikely]]
        {
            if (!computeExecutionOrder()) [[unlikely]]
            {
                return STATUS(EFAULT, "Failed to compute execution order for the DAG");
            }
        }

        for (auto *node: _execution_order)
        {
            if (const auto &status = node->operator()(); !status) [[unlikely]]
            {
                return status;
            }
        }

        return STATUS_OK();
    }

    inline core::Status operator()(core::Reporter &reporter) noexcept
    {
        if (_execution_order.empty()) [[unlikely]]
        {
            if (!computeExecutionOrder()) [[unlikely]]
            {
                return STATUS(EFAULT, "Failed to compute execution order for the DAG");
            }
        }

        for (auto *node: _execution_order)
        {
            if (const auto &status = node->operator()(reporter); !status) [[unlikely]]
            {
                return status;
            }
        }

        return STATUS_OK();
    }

private:
    const std::string_view _name;
    std::forward_list<std::shared_ptr<module::MNode>> _nodes;
    std::unordered_map<MNode *, std::forward_list<MNode *>> _adj;
    std::unordered_map<MNode *, int> _in_degree;
    mutable std::vector<MNode *> _execution_order;
};

} // namespace module

namespace bridge {

extern void __REGISTER_MODULE_DAG_BUILDER__();

} // namespace bridge

#endif
