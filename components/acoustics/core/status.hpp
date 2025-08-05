#pragma once
#ifndef STATUS_HPP
#define STATUS_HPP

#include "logger.hpp"

#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#ifdef STATUS
#warning "STATUS macro is already defined, using core::Status instead"
#else
#if LOG_LEVEL > ERROR
#define STATUS(code, message) core::Status(code, message)
#else
#define STATUS(code, message) core::Status(code)
#endif
#define STATUS_CODE(code) core::Status(code)
#define STATUS_OK()       core::Status::OK()
#endif

namespace core {

class Status final
{
public:
    static inline constexpr Status OK() noexcept
    {
        return Status(0);
    }

    inline constexpr Status(int code = 0, std::string &&message = "") noexcept
        : _code(code), _message(std::move(message))
    {
        if (_code != 0 && _message.empty()) [[unlikely]]
        {
            _message = std::strerror(code);
        }
    }

    inline Status(const Status &other) noexcept : _code(other._code), _message(other._message) { }

    inline Status(Status &&other) noexcept : _code(other._code), _message(std::move(other._message))
    {
        other._code = 0;
        other._message.clear();
    }

    ~Status() noexcept = default;

    inline operator bool() const noexcept
    {
        return _code == 0;
    }

    inline int code() const noexcept
    {
        return _code;
    }

    const std::string &message() const noexcept
    {
        return _message;
    }

    inline Status &operator=(const Status &other) noexcept
    {
        if (this != &other)
        {
            _code = other._code;
            _message = other._message;
        }
        return *this;
    }

    inline Status &operator=(Status &&other) noexcept
    {
        if (this != &other)
        {
            _code = other._code;
            _message = std::move(other._message);
        }
        return *this;
    }

private:
    int _code;
    std::string _message;
};

} // namespace core

#endif