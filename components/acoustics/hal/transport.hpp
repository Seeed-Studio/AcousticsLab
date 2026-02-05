#pragma once
#ifndef TRANSPORT_HPP
#define TRANSPORT_HPP

#include "core/config_object.hpp"
#include "core/data_frame.hpp"
#include "core/logger.hpp"
#include "core/ring_buffer.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <unordered_map>

namespace hal {

class Transport;

class TransportRegistry final
{
public:
    using TransportMap = std::unordered_map<int, Transport *>;

    TransportRegistry() = default;
    ~TransportRegistry() = default;

    inline static Transport *getTransport(int id) noexcept
    {
        auto it = _transports.find(id);
        if (it != _transports.end())
        {
            return it->second;
        }
        return nullptr;
    }

    const TransportMap &getTransports() const noexcept
    {
        return _transports;
    }

    static const TransportMap &getTransportMap() noexcept
    {
        return _transports;
    }

protected:
    friend class Transport;

    static core::Status registerTransport(Transport *transport) noexcept;

private:
    static TransportMap _transports;
};

class Transport
{
public:
    enum class Type {
        Unknown = 0,
        UART,
        I2C,
        SPI,
        MQTT,
    };

    enum class Status : size_t {
        Unknown = 0,
        Uninitialized,
        Idle,
        Locked,
    };

    struct Info final
    {
        Info(int id, std::string_view name, Type type, core::ConfigObjectMap &&configs) noexcept
            : id(id), name(name), type(type), status(Status::Unknown), configs(std::move(configs))
        {
        }

        ~Info() = default;

        const int id;
        const std::string_view name;
        const Type type;
        volatile Status status;
        core::ConfigObjectMap configs;
    };

    virtual ~Transport() = default;

    virtual core::Status init() noexcept = 0;
    virtual core::Status deinit() noexcept = 0;

    virtual inline bool initialized() const noexcept
    {
        return _info.status >= Status::Idle;
    }

    const Info &info() const noexcept
    {
        syncInfo(_info);
        return _info;
    }

    virtual core::Status updateConfig(const core::ConfigMap &configs) noexcept = 0;

    virtual inline size_t available() const noexcept = 0;
    virtual inline size_t read(void *data, size_t size) noexcept = 0;
    virtual inline size_t write(const void *data, size_t size) noexcept = 0;

    virtual inline int flush() noexcept = 0;

    using ReadCallback = std::function<void(core::RingBuffer<std::byte> &, size_t)>;
    using ReadCondition = std::function<bool(const core::RingBuffer<std::byte> &)>;

    core::Status readIf(ReadCallback callback, ReadCondition condition) noexcept
    {
        if (!initialized()) [[unlikely]]
        {
            return STATUS(EFAULT, "Transport is not initialized");
        }
        const auto ret = syncReadBuffer();
        if (ret != 0) [[unlikely]]
        {
            LOG(ERROR, "Failed to sync read buffer: %s", std::strerror(ret));
        }
        auto &buffer = getReadBuffer();
        if (condition ? condition(buffer) : true)
        {
            if (callback) [[likely]]
            {
                callback(buffer, buffer.tellg());
            }
        }
        return STATUS_OK();
    }

protected:
    Transport(Info &&info) noexcept : _info(std::move(info))
    {
        LOG(DEBUG, "Registering transport: ID=%d, Name='%s', Type=%d", _info.id, _info.name.data(),
            static_cast<int>(_info.type));
        [[maybe_unused]] auto status = TransportRegistry::registerTransport(this);
        if (!status)
        {
            LOG(ERROR, "Failed to register transport: %s", status.message().c_str());
        }
        _info.status = Status::Uninitialized;
        LOG(DEBUG, "Transport '%s' registered successfully", _info.name.data());
    }

    virtual void syncInfo(Info &info) const noexcept { }

    virtual inline int syncReadBuffer() const noexcept
    {
        return ENOTSUP;
    }

    virtual inline core::RingBuffer<std::byte> &getReadBuffer() noexcept = 0;

    mutable Info _info;
};

} // namespace hal

namespace bridge {

extern void __REGISTER_TRANSPORTS__();

}

#endif // TRANSPORT_HPP
