#pragma once
#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "hal/device.hpp"
#include "hal/engine.hpp"
#include "hal/sensor.hpp"
#include "hal/transport.hpp"

#include "core/logger.hpp"
#include "core/status.hpp"

#include "module/module_builder.hpp"
#include "module/module_dag.hpp"
#include "module/module_node.hpp"

#include <mutex>
#include <string_view>

namespace api {

class Context final
{
public:
    ~Context()
    {
        LOG(INFO, "API Context destroyed");
    }

    inline std::string_view version() const noexcept
    {
        return _version;
    }

    inline core::Status status() const noexcept
    {
        return _status;
    }

    inline hal::Transport *console() const noexcept
    {
        return _console_ptr;
    }

    inline Context *getptr() noexcept
    {
        return this;
    }

    inline void *context() noexcept
    {
        return _user_context;
    }

    virtual core::Status report(core::Status status, hal::Transport *transport = nullptr) noexcept;

    static Context *create() noexcept;

protected:
    Context(std::string_view version, int console, void *user_context) noexcept
        : _version(version), _console(console), _console_ptr(nullptr), _status(STATUS_OK()), _user_context(user_context)
    {
        static core::Status init_result = init(_console, _console_ptr);
        _status = init_result;
        if (!_status) [[unlikely]]
        {
            LOG(ERROR, "Failed to initialize API Context: %s", _status.message().c_str());
        }
        if (!_user_context)
        {
            LOG(WARNING, "User context is not set, using nullptr");
        }

        LOG(INFO, "API Context version %s initialized successfully", _version.data());
    }

private:
    static core::Status init(int console, hal::Transport *&console_ptr) noexcept
    {
        auto status = STATUS_OK();

        bridge::__REGISTER_DEVICE__();
        {
            auto device = hal::DeviceRegistry::getDevice();
            if (!device) [[unlikely]]
            {
                return STATUS(ENODEV, "Default device is not registered");
            }
            if (!device->initialized()) [[likely]]
            {
                status = device->init();
                if (!status) [[unlikely]]
                {
                    return status;
                }
            }
        }

        bridge::__REGISTER_ENGINES__();

        bridge::__REGISTER_SENSORS__();

        bridge::__REGISTER_TRANSPORTS__();
        {
            const auto &transports = hal::TransportRegistry::getTransportMap();
            for (const auto &[id, transport]: transports)
            {
                if (transport && !transport->initialized()) [[likely]]
                {
                    status = transport->init();
                    if (!status) [[unlikely]]
                    {
                        LOG(ERROR, "Failed to initialize transport ID=%d: %s", id, status.message().c_str());
                    }
                }
                if (!console_ptr && id == console)
                {
                    console_ptr = transport;
                }
            }
            if (!console_ptr || !console_ptr->initialized()) [[unlikely]]
            {
                return STATUS(ENODEV, "Transport with ID " + std::to_string(console) + " is not registered");
            }
        }

        bridge::__REGISTER_PREDEFINED_MODULE_NODE_BUILDER__();
        bridge::__REGISTER_INTERNAL_MODULE_NODE_BUILDER__();
        bridge::__REGISTER_EXTERNAL_MODULE_NODE_BUILDER__();

        bridge::__REGISTER_MODULE_DAG_BUILDER__();

        return status;
    }

    const std::string_view _version;
    const int _console;
    hal::Transport *_console_ptr;
    core::Status _status;
    void *_user_context;
};

} // namespace api

#endif // CONTEXT_HPP
