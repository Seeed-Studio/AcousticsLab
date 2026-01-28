#pragma once
#ifndef SERVER_HPP
#define SERVER_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"

#include "hal/transport.hpp"

#include "command.hpp"
#include "context.hpp"
#include "executor.hpp"
#include "parser.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <variant>

namespace api {

class Server final
{
public:
    using StopToken = std::function<bool(const core::Status &)>;

    Server(Context &context, Executor &executor, size_t buffer_size = 2048) noexcept
        : _context(context), _executor(executor), _transports(hal::TransportRegistry::getTransportMap()),
          _commands(CommandRegistry::getCommandMap()), _buffer_size(buffer_size), _buffer(nullptr),
          _read_status(STATUS_OK())
    {
        if (_buffer_size < 1024) [[unlikely]]
        {
            LOG(ERROR, "Buffer size must be at least 1024 bytes");
            return;
        }
        _buffer = new unsigned char[_buffer_size];
        if (!_buffer) [[unlikely]]
        {
            LOG(ERROR, "Failed to allocate buffer of size %zu", _buffer_size);
            return;
        }

        std::erase_if(_transports, [](const auto &pair) {
            if (!pair.second) [[unlikely]]
            {
                LOG(WARNING, "Removing transport with ID %d due to null pointer", pair.first);
                return true;
            }
            else if (!pair.second->initialized()) [[unlikely]]
            {
                LOG(WARNING, "Removing transport with ID %d due to uninitialized state", pair.first);
                return true;
            }
            return false;
        });
        if (_transports.empty()) [[unlikely]]
        {
            LOG(WARNING, "No transports registered in API Server");
        }

        if (_commands.empty()) [[unlikely]]
        {
            LOG(WARNING, "No commands registered in API Server");
        }

        const auto id = executor.id();
        for (auto &cmd: _commands)
        {
            if (!cmd.second) [[unlikely]]
            {
                LOG(WARNING, "Command '%s' is null, skipping", cmd.first.data());
                continue;
            }
            auto rc = cmd.second->rc();
            if (rc == nullptr)
            {
                continue;
            }
            auto status = rc(_context, _executor, id);
            if (!status) [[unlikely]]
            {
                LOG(ERROR, "RC failed on command '%s': %s", cmd.second->name().data(), status.message().c_str());
                _context.report(status);
            }
        }

        LOG(INFO, "API Server created with buffer size: %zu", _buffer_size);
    }

    ~Server() noexcept
    {
        if (_buffer)
        {
            delete[] _buffer;
            _buffer = nullptr;
        }

        LOG(INFO, "API Server destroyed");
    }

    inline core::Status serve(StopToken stop_token) noexcept
    {
        if (!stop_token) [[unlikely]]
        {
            return STATUS(EINVAL, "Stop token is null");
        }
        if (_transports.empty()) [[unlikely]]
        {
            return STATUS(ENODEV, "No transports registered in API Server");
        }
        if (!_buffer) [[unlikely]]
        {
            return STATUS(EFAULT, "API Server buffer is null");
        }

        while (!stop_token(_read_status))
        {
            for (const auto &[id, transport]: _transports)
            {
                _read_status = transport->readIf(std::bind(&Server::transportReadCallback, this, std::ref(*transport),
                                                     std::placeholders::_1, std::placeholders::_2),
                    std::bind(&Server::transportReadCondition, this, std::placeholders::_1));
                if (!_read_status) [[unlikely]]
                {
                    LOG(DEBUG, "Transport ID=%d read status: %s", id, _read_status.message().c_str());
                    if (stop_token(_read_status))
                    {
                        return _read_status;
                    }
                }
            }
        }

        return STATUS_OK();
    }

protected:
    void transportReadCallback(hal::Transport &transport, core::RingBuffer<std::byte> &buffer, size_t size) noexcept
    {
        if (size >= _buffer_size) [[unlikely]]
        {
            LOG(WARNING, "Transport read callback called with size %zu, exceeding buffer size %zu", size, _buffer_size);
            size = _buffer_size - 1;
        }
        else if (size == 0) [[unlikely]]
        {
            LOG(WARNING, "Transport read callback called with zero size");
            return;
        }
        const size_t read = buffer.read(reinterpret_cast<std::byte *>(_buffer), size);
        if (read <= 1) [[unlikely]]
        {
            return;
        }
        std::string_view data(reinterpret_cast<const char *>(_buffer), read - 1);
        Command *command_ptr = nullptr;
        auto config_objects = core::ConfigMap();
        _read_status = Parser::parseCommand(data, this->_commands, command_ptr, config_objects);
        if (!_read_status || !command_ptr)
        {
            _buffer[read - 1] = '\0';
            LOG(DEBUG, "Failed to parse command '%s': %s", reinterpret_cast<const char *>(_buffer),
                _read_status.message().c_str());
            _context.report(_read_status, &transport);
            return;
        }
        LOG(DEBUG, "Parsed command: '%s'", command_ptr->name().data());
        auto task = command_ptr->operator()(_context, transport, config_objects, _executor.id());
        _read_status = _executor.submit(task);
    }

    bool transportReadCondition(const core::RingBuffer<std::byte> &buffer) const noexcept
    {
        if (buffer.empty())
        {
            return false;
        }
        std::byte val;
        while (buffer.peek(val))
        {
            if (val == std::byte('\n') || val == std::byte('\r'))
            {
                return true;
            }
        }
        return false;
    }

private:
    Context &_context;
    Executor &_executor;
    hal::TransportRegistry::TransportMap _transports;
    const CommandRegistry::CommandMap &_commands;
    const size_t _buffer_size;
    unsigned char *_buffer;
    core::Status _read_status;
};

} // namespace api

#endif // SERVER_HPP
