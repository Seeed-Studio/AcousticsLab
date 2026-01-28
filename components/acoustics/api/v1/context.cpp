#include "common.hpp"

#include "cmd_break.hpp"
#include "cmd_cfgsc.hpp"
#include "cmd_info.hpp"
#include "cmd_rst.hpp"
#include "cmd_start.hpp"
#include "cmd_ver.hpp"

#include "api/context.hpp"

#include "core/logger.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace v1 {

static bool initContext()
{
    static auto serializer = core::Serializer::create('\r');
    if (!serializer) [[unlikely]]
    {
        LOG(ERROR, "Failed to create default serializer");
        return false;
    }
    v1::defaults::serializer = serializer.get();

    return true;
}

static bool preInitHooks()
{
    v1::CmdCfgSC::preInitHook();
    v1::CmdStart::preInitHook();

    return true;
}

static bool registerCommands()
{
    [[maybe_unused]] static auto break_cmd = v1::CmdBreak();
    [[maybe_unused]] static auto rst_cmd = v1::CmdRst();
    [[maybe_unused]] static auto ver_cmd = v1::CmdVer();
    [[maybe_unused]] static auto info_cmd = v1::CmdInfo();
    [[maybe_unused]] static auto cfgsc_cmd = v1::CmdCfgSC();
    [[maybe_unused]] static auto start_cmd = v1::CmdStart();

    return true;
}

} // namespace v1

namespace api {

core::Status Context::report(core::Status status, hal::Transport *transport) noexcept
{
    if (!transport)
    {
        transport = _console_ptr;
    }
    if (!transport) [[unlikely]]
    {
        LOG(ERROR, "Console transport is not initialized");
        return status;
    }

    static size_t counter = 0;

    auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
        std::bind(&v1::defaults::write_callback, std::ref(*transport), std::placeholders::_1, std::placeholders::_2),
        std::bind(&v1::defaults::flush_callback, std::ref(*transport)));

    writer["type"] += v1::ResponseType::System;
    writer["name"] += "SYSTEM";
    writer["tag"] += std::to_string(counter++);
    writer["code"] += status.code();
    writer["msg"] << status.message();
    {
        auto data = writer["data"].writer<core::ObjectWriter>();
        data["ts"]
            += std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                   .count();
    }

    return STATUS_OK();
}

Context *Context::create() noexcept
{
    static Context instance(CONTEXT_VERSION, CONTEXT_CONSOLE, nullptr);

    if (!v1::initContext())
    {
        return nullptr;
    }

    if (!v1::preInitHooks())
    {
        return nullptr;
    }

    if (!v1::registerCommands())
    {
        return nullptr;
    }

    return &instance;
}

} // namespace api
