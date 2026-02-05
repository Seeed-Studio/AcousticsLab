#pragma once
#ifndef COMMON_HPP
#define COMMON_HPP

#include "core/ring_buffer.hpp"
#include "core/serializer.hpp"

#include "hal/device.hpp"
#include "hal/transport.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

#ifdef CONTEXT_PREFIX
#error "CONTEXT_PREFIX is already defined. Please ensure it is defined only once."
#else
#define CONTEXT_PREFIX "api/"
#endif

#ifdef CONTEXT_VERSION
#error "CONTEXT_VERSION is already defined. Please ensure it is defined only once."
#else
#define CONTEXT_VERSION "1"
#endif

#ifdef CONTEXT_CONSOLE
#error "CONTEXT_CONSOLE is already defined. Please ensure it is defined only once."
#else
#define CONTEXT_CONSOLE 0
#endif

namespace v1 {

namespace defaults {

    inline constexpr const std::string_view at_version = "1.8.27";
    inline constexpr const std::string_view firmware_type = "sound_classification";
    inline constexpr const size_t task_priority = 5;
    inline constexpr const int engine_id = 1;
    inline core::Serializer *serializer = nullptr;

    inline void wait_callback() noexcept
    {
        hal::DeviceRegistry::getDevice()->sleep(5);
    }

    inline int write_callback(hal::Transport &transport, const void *data, size_t size) noexcept
    {
        return transport.write(data, size);
    }

    inline int flush_callback(hal::Transport &transport) noexcept
    {
        transport.write("\n", 1);
        return transport.flush();
    }

} // namespace defaults

struct ResponseType final
{
    static constexpr const int Direct = 0;
    static constexpr const int Event = 1;
    static constexpr const int Stream = 2;
    static constexpr const int System = 3;
    static constexpr const int Unknown = 4;
};

} // namespace v1

#endif
