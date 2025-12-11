#pragma once
#ifndef CMD_CFGSC_HPP
#define CMD_CFGSC_HPP

#include "common.hpp"
#include "task_sc.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/logger.hpp"

#include "hal/device.hpp"
#include "hal/sensor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace v1 {

class CmdCfgSC final: public api::Command
{
    static inline constexpr const char *overlap_ratio_path = CONTEXT_PREFIX CONTEXT_VERSION "/overlap_ratio";
    static inline constexpr const char *rms_normalize_path = CONTEXT_PREFIX CONTEXT_VERSION "/rms_normalize";
    static inline constexpr const char *threshold_path = CONTEXT_PREFIX CONTEXT_VERSION "/threshold";

public:
    static inline void preInitHook()
    {
        auto device = hal::DeviceRegistry::getDevice();
        if (!device || !device->initialized()) [[unlikely]]
        {
            LOG(ERROR, "Device not registered or not initialized");
            return;
        }

        {
            float overlap_ratio = shared::overlap_ratio.load();
            size_t size = sizeof(overlap_ratio);
            auto status = device->load(hal::Device::StorageType::Internal, overlap_ratio_path, &overlap_ratio, size);
            if (!status || size != sizeof(overlap_ratio)) [[unlikely]]
            {
                LOG(ERROR, "Failed to load overlap_ratio or size mismatch");
                return;
            }
            shared::overlap_ratio = std::clamp(overlap_ratio, shared::overlap_ratio_min, shared::overlap_ratio_max);
        }

        {
            bool rms_normalize = shared::rms_normalize.load();
            size_t size = sizeof(rms_normalize);
            auto status = device->load(hal::Device::StorageType::Internal, rms_normalize_path, &rms_normalize, size);
            if (!status || size != sizeof(rms_normalize)) [[unlikely]]
            {
                LOG(ERROR, "Failed to load rms_normalize or size mismatch");
                return;
            }
            shared::rms_normalize = rms_normalize;
        }

        {
            float threshold = shared::threshold.load();
            size_t size = sizeof(threshold);
            auto status = device->load(hal::Device::StorageType::Internal, threshold_path, &threshold, size);
            if (!status || size != sizeof(threshold)) [[unlikely]]
            {
                LOG(ERROR, "Failed to load threshold or size mismatch");
                return;
            }
            shared::threshold = std::clamp(threshold, shared::threshold_min, shared::threshold_max);
        }
    }

    CmdCfgSC()
        : Command("CFGSC", "Configure the Sound Classification",
              core::ConfigObjectMap {
                  CONFIG_OBJECT_DECL_FLOAT("overlap_ratio", "Overlap ratio of sound samples for each classification",
                      v1::shared::overlap_ratio.load(), shared::overlap_ratio_min, shared::overlap_ratio_max),
                  CONFIG_OBJECT_DECL_BOOLEAN("rms_normalize", "Enable or disable RMS normalization",
                      v1::shared::rms_normalize.load()),
                  CONFIG_OBJECT_DECL_FLOAT("threshold", "Confidence threshold for sound classification",
                      v1::shared::threshold.load(), shared::threshold_min, shared::threshold_max),
              }),
          _device(hal::DeviceRegistry::getDevice())
    {
    }

    ~CmdCfgSC() noexcept override = default;

    std::shared_ptr<api::Task> operator()(api::Context &context, hal::Transport &transport, const core::ConfigMap &args,
        size_t id) override
    {
        auto status = STATUS_OK();

        if (auto it = args.find("@0"); it != args.end())
        {
            if (auto overlap_ratio_str = std::get_if<std::string>(&it->second);
                overlap_ratio_str != nullptr && !overlap_ratio_str->empty())
            {
                auto &target = _args["overlap_ratio"];
                if ((status = target.setValue(*overlap_ratio_str)))
                {
                    const auto old = shared::overlap_ratio.load();
                    float overlap_ratio = target.getValue<float>();
                    if (old != overlap_ratio && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, overlap_ratio_path,
                                 &overlap_ratio, sizeof(overlap_ratio))))
                    {
                        goto Reply;
                    }
                    v1::shared::overlap_ratio = overlap_ratio;
                }
                else
                    goto Reply;
            }
        }

        if (auto it = args.find("@1"); it != args.end())
        {
            if (auto rms_normalize_str = std::get_if<std::string>(&it->second);
                rms_normalize_str != nullptr && !rms_normalize_str->empty())
            {
                auto &target = _args["rms_normalize"];
                if ((status = target.setValue(*rms_normalize_str)))
                {
                    const auto old = shared::rms_normalize.load();
                    bool rms_normalize = target.getValue<bool>();
                    if (old != rms_normalize && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, rms_normalize_path,
                                 &rms_normalize, sizeof(rms_normalize))))
                    {
                        goto Reply;
                    }
                    v1::shared::rms_normalize = rms_normalize;
                }
                else
                    goto Reply;
            }
        }

        if (auto it = args.find("@2"); it != args.end())
        {
            if (auto threshold_str = std::get_if<std::string>(&it->second);
                threshold_str != nullptr && !threshold_str->empty())
            {
                auto &target = _args["threshold"];
                if ((status = target.setValue(*threshold_str)))
                {
                    const auto old = shared::threshold.load();
                    float threshold = target.getValue<float>();
                    if (old != threshold && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, threshold_path, &threshold,
                                 sizeof(threshold))))
                    {
                        goto Reply;
                    }
                    v1::shared::threshold = threshold;
                }
                else
                    goto Reply;
            }
        }

    Reply: {
        auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
            std::bind(&v1::defaults::write_callback, std::ref(transport), std::placeholders::_1, std::placeholders::_2),
            std::bind(&v1::defaults::flush_callback, std::ref(transport)));

        writer["type"] += v1::ResponseType::Direct;
        writer["name"] += _name;
        if (auto it = args.find("@cmd_tag"); it != args.end())
        {
            auto tag = std::get_if<std::string>(&it->second);
            if (tag != nullptr)
                writer["tag"] << *tag;
        }
        writer["code"] += status.code();
        writer["msg"] << status.message();
        {
            auto data = writer["data"].writer<core::ObjectWriter>();
            {
                auto args = data["args"].writer<core::ArrayWriter>();
                {
                    auto arg = args.writer<core::ArrayWriter>();
                    arg += shared::overlap_ratio.load();
                    arg += shared::overlap_ratio_min;
                    arg += shared::overlap_ratio_max;
                }
                {
                    auto arg = args.writer<core::ArrayWriter>();
                    arg += shared::rms_normalize.load();
                    arg += 0;
                    arg += 1;
                }
                {
                    auto arg = args.writer<core::ArrayWriter>();
                    arg += shared::threshold.load();
                    arg += shared::threshold_min;
                    arg += shared::threshold_max;
                }
            }
            data["sampleChunkMs"] += shared::sample_chunk_ms;
        }
    }

        return nullptr;
    }

private:
    hal::Device *_device;
};

} // namespace v1

#endif // CMD_CFGSC_HPP
