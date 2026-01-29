#pragma once
#ifndef SERIALIZER_HPP
#define SERIALIZER_HPP

#include "logger.hpp"
#include "traits.hpp"

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace core {

namespace charconv {

    namespace digits2 {

        inline constexpr decltype(auto) make_digits() noexcept
        {
            alignas(2) std::array<char, 200> data {};
            for (size_t i = 0; i < 100; ++i)
            {
                data[i * 2] = static_cast<char>('0' + i / 10);
                data[i * 2 + 1] = static_cast<char>('0' + i % 10);
            }
            return data;
        }

        alignas(2) inline constexpr auto table = make_digits();

        inline constexpr const char *at(size_t index)
        {
            return &table[index];
        }

    } // namespace digits2

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_unsigned_v<U> && sizeof(U) == sizeof(uint8_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
        if (val < 100U)
        {
            unsigned leading_zero = val < 10U;
            std::memcpy(buf, digits2::at((val << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
            return buf + 2 - leading_zero;
#pragma GCC diagnostic pop
        }
        else
        {
            uint16_t l0_0 = (41U * val) >> 12;
            uint16_t l1_2 = val - (100U * l0_0);
            *buf = static_cast<char>('0' | l0_0);
            std::memcpy(buf + 1, digits2::at(l1_2 << 1), 2);
            return buf + 3;
        }
#pragma GCC diagnostic pop
    }

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_signed_v<U> && sizeof(U) == sizeof(int8_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
        unsigned sign = val < 0;
        uint8_t uval = static_cast<uint8_t>(val);
        *buf = '-';
        return itoa(sign ? static_cast<uint8_t>(~uval + 1) : uval, buf + sign);
    }

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_unsigned_v<U> && sizeof(U) == sizeof(uint16_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
        if (val < 100U)
        {
            unsigned leading_zero = val < 10U;
            std::memcpy(buf, digits2::at((val << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
            return buf + 2 - leading_zero;
#pragma GCC diagnostic pop
        }
        else if (val < 10'000U)
        {
            uint32_t l0_1 = (static_cast<uint32_t>(5243U) * val) >> 19;
            uint32_t l2_3 = val - (static_cast<uint32_t>(100U) * l0_1);
            unsigned leading_zero = l0_1 < 10U;
            std::memcpy(buf, digits2::at((l0_1 << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
            buf -= leading_zero;
#pragma GCC diagnostic pop
            std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
            return buf + 4;
        }
        else
        {
            uint32_t l0_1 = static_cast<uint32_t>((static_cast<uint64_t>(107375UL) * val) >> 30);
            uint32_t l2_5 = val - (static_cast<uint32_t>(10'000U) * l0_1);
            uint32_t l2_3 = (static_cast<uint32_t>(5243U) * l2_5) >> 19;
            uint32_t l4_5 = l2_5 - (static_cast<uint32_t>(100U) * l2_3);
            unsigned leading_zero = l0_1 < 10U;
            std::memcpy(buf, digits2::at((l0_1 << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
            buf -= leading_zero;
#pragma GCC diagnostic pop
            std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
            std::memcpy(buf + 4, digits2::at(l4_5 << 1), 2);
            return buf + 6;
        }
#pragma GCC diagnostic pop
    }

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_signed_v<U> && sizeof(U) == sizeof(int16_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
        unsigned sign = val < 0;
        uint16_t uval = static_cast<uint16_t>(val);
        *buf = '-';
        return itoa(sign ? static_cast<uint16_t>(~uval + 1) : uval, buf + sign);
    }

    inline char *itoa_lt_1e2(uint32_t val, char *buf) noexcept
    {
        unsigned leading_zero = val < 10U;
        std::memcpy(buf, digits2::at((val << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        return buf + 2 - leading_zero;
#pragma GCC diagnostic pop
    }

    inline char *itoa_ge_1e2_lt_1e4(uint32_t val, char *buf) noexcept
    {
        uint32_t l0_1 = (static_cast<uint32_t>(5243U) * val) >> 19;
        uint32_t l2_3 = val - (static_cast<uint32_t>(100U) * l0_1);
        unsigned leading_zero = l0_1 < 10U;
        std::memcpy(buf, digits2::at((l0_1 << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        buf -= leading_zero;
#pragma GCC diagnostic pop
        std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
        return buf + 4;
    }

    inline char *itoa_ge_1e2_lt_1e4_lz(uint32_t val, char *buf) noexcept
    {
        uint32_t l0_1 = (static_cast<uint32_t>(5243U) * val) >> 19;
        uint32_t l2_3 = val - (static_cast<uint32_t>(100U) * l0_1);
        std::memcpy(buf, digits2::at(l0_1 << 1), 2);
        std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
        return buf + 4;
    }

    inline char *itoa_ge_1e4_lt_1e6(uint32_t val, char *buf) noexcept
    {
        uint32_t l0_1 = static_cast<uint32_t>((static_cast<uint64_t>(429497UL) * val) >> 32);
        uint32_t l2_5 = val - (static_cast<uint32_t>(10'000U) * l0_1);
        uint32_t l2_3 = (static_cast<uint32_t>(5243U) * l2_5) >> 19;
        uint32_t l4_5 = l2_5 - (static_cast<uint32_t>(100U) * l2_3);
        unsigned leading_zero = l0_1 < 10U;
        std::memcpy(buf, digits2::at((l0_1 << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        buf -= leading_zero;
#pragma GCC diagnostic pop
        std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
        std::memcpy(buf + 4, digits2::at(l4_5 << 1), 2);
        return buf + 6;
    }

    inline char *itoa_ge_1e6_lt_1e8(uint32_t val, char *buf) noexcept
    {
        uint32_t l0_3 = static_cast<uint32_t>((static_cast<uint64_t>(109951163UL) * val) >> 40);
        uint32_t l4_7 = val - (static_cast<uint32_t>(10'000U) * l0_3);
        uint32_t l0_1 = (static_cast<uint32_t>(5243U) * l0_3) >> 19;
        uint32_t l2_3 = l0_3 - (static_cast<uint32_t>(100U) * l0_1);
        uint32_t l4_5 = (static_cast<uint32_t>(5243U) * l4_7) >> 19;
        uint32_t l6_7 = l4_7 - (static_cast<uint32_t>(100U) * l4_5);
        unsigned leading_zero = l0_1 < 10U;
        std::memcpy(buf, digits2::at((l0_1 << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        buf -= leading_zero;
#pragma GCC diagnostic pop
        std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
        std::memcpy(buf + 4, digits2::at(l4_5 << 1), 2);
        std::memcpy(buf + 6, digits2::at(l6_7 << 1), 2);
        return buf + 8;
    }

    inline char *itoa_ge_1e6_lt_1e8_lz(uint32_t val, char *buf) noexcept
    {
        uint32_t l0_3 = static_cast<uint32_t>((static_cast<uint64_t>(109951163UL) * val) >> 40);
        uint32_t l4_7 = val - (static_cast<uint32_t>(10'000U) * l0_3);
        uint32_t l0_1 = (static_cast<uint32_t>(5243U) * l0_3) >> 19;
        uint32_t l2_3 = l0_3 - (static_cast<uint32_t>(100U) * l0_1);
        uint32_t l4_5 = (static_cast<uint32_t>(5243U) * l4_7) >> 19;
        uint32_t l6_7 = l4_7 - (static_cast<uint32_t>(100U) * l4_5);
        std::memcpy(buf, digits2::at((l0_1 << 1)), 2);
        std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
        std::memcpy(buf + 4, digits2::at(l4_5 << 1), 2);
        std::memcpy(buf + 6, digits2::at(l6_7 << 1), 2);
        return buf + 8;
    }

    inline char *itoa_ge_1e8_lt_1ea(uint32_t val, char *buf) noexcept
    {
        uint32_t l0_5 = static_cast<uint32_t>((static_cast<uint64_t>(3518437209UL) * val) >> 45);
        uint32_t l0_1 = static_cast<uint32_t>((static_cast<uint64_t>(429497UL) * l0_5) >> 32);
        uint32_t l2_5 = l0_5 - (static_cast<uint32_t>(10'000U) * l0_1);
        uint32_t l2_3 = (static_cast<uint32_t>(5243U) * l2_5) >> 19;
        uint32_t l4_5 = l2_5 - (static_cast<uint32_t>(100U) * l2_3);
        uint32_t l6_9 = val - (static_cast<uint32_t>(10'000U) * l0_5);
        uint32_t l6_7 = (static_cast<uint32_t>(5243U) * l6_9) >> 19;
        uint32_t l8_9 = l6_9 - (static_cast<uint32_t>(100U) * l6_7);
        unsigned leading_zero = l0_1 < 10U;
        std::memcpy(buf, digits2::at((l0_1 << 1) + leading_zero), 2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        buf -= leading_zero;
#pragma GCC diagnostic pop
        std::memcpy(buf + 2, digits2::at(l2_3 << 1), 2);
        std::memcpy(buf + 4, digits2::at(l4_5 << 1), 2);
        std::memcpy(buf + 6, digits2::at(l6_7 << 1), 2);
        std::memcpy(buf + 8, digits2::at(l8_9 << 1), 2);
        return buf + 10;
    }

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_unsigned_v<U> && sizeof(U) == sizeof(uint32_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
        if (val < 100U)
        {
            return itoa_lt_1e2(val, buf);
        }
        else if (val < 10'000U)
        {
            return itoa_ge_1e2_lt_1e4(val, buf);
        }
        else if (val < static_cast<uint32_t>(1'000'000UL))
        {
            return itoa_ge_1e4_lt_1e6(val, buf);
        }
        else if (val < static_cast<uint32_t>(100'000'000UL))
        {
            return itoa_ge_1e6_lt_1e8(val, buf);
        }
        else
        {
            return itoa_ge_1e8_lt_1ea(val, buf);
        }
    }

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_signed_v<U> && sizeof(U) == sizeof(int32_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
        unsigned sign = val < 0;
        uint32_t uval = static_cast<uint32_t>(val);
        *buf = '-';
        return itoa(sign ? static_cast<uint32_t>(~uval + 1) : uval, buf + sign);
    }

    inline char *itoa_lt_1e8(uint32_t val, char *buf) noexcept
    {
        if (val < 100U)
        {
            return itoa_lt_1e2(val, buf);
        }
        else if (val < 10'000U)
        {
            return itoa_ge_1e2_lt_1e4(val, buf);
        }
        else if (val < static_cast<uint32_t>(1'000'000UL))
        {
            return itoa_ge_1e4_lt_1e6(val, buf);
        }
        else
        {
            return itoa_ge_1e6_lt_1e8(val, buf);
        }
    }

    inline char *itoa_ge_1e4_lt_1e8(uint32_t val, char *buf) noexcept
    {
        if (val < static_cast<uint32_t>(1'000'000UL))
        {
            return itoa_ge_1e4_lt_1e6(val, buf);
        }
        else
        {
            return itoa_ge_1e6_lt_1e8(val, buf);
        }
    }

    inline char *itoa_ge_1e8_lt_1e16(uint64_t val, char *buf) noexcept
    {
        uint64_t l0_7 = val / 100'000'000UL;
        uint32_t l8_f = static_cast<uint32_t>(val - (static_cast<uint64_t>(100'000'000UL) * l0_7));
        buf = itoa_lt_1e8(static_cast<uint32_t>(l0_7), buf);
        buf = itoa_ge_1e6_lt_1e8_lz(l8_f, buf);
        return buf;
    }

    inline char *itoa_ge_1e16_lt_1e20(uint64_t val, char *buf) noexcept
    {
        uint64_t l0_b = val / 100'000'000UL;
        uint32_t r0_7 = static_cast<uint32_t>(val - (static_cast<uint64_t>(100'000'000UL) * l0_b));
        uint32_t l0_3 = static_cast<uint32_t>(l0_b / 10'000UL);
        uint32_t l4_7 = static_cast<uint32_t>(l0_b - (static_cast<uint32_t>(10'000UL) * l0_3));
        buf = itoa_ge_1e4_lt_1e8(l0_3, buf);
        buf = itoa_ge_1e2_lt_1e4_lz(l4_7, buf);
        buf = itoa_ge_1e6_lt_1e8_lz(r0_7, buf);
        return buf;
    }

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_unsigned_v<U> && sizeof(U) == sizeof(uint64_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
        if (val < static_cast<uint64_t>(100'000'000UL))
        {
            return itoa_lt_1e8(static_cast<uint32_t>(val), buf);
        }
        else if (val < static_cast<uint64_t>(10'000'000'000'000'000ULL))
        {
            return itoa_ge_1e8_lt_1e16(val, buf);
        }
        else
        {
            return itoa_ge_1e16_lt_1e20(val, buf);
        }
    }

    template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
        std::enable_if_t<std::is_signed_v<U> && sizeof(U) == sizeof(int64_t), bool> = true>
    inline constexpr char *itoa(T &&val, char *buf) noexcept
    {
        unsigned sign = val < 0;
        uint64_t uval = static_cast<uint64_t>(val);
        *buf = '-';
        return itoa(sign ? static_cast<uint64_t>(~uval + 1) : uval, buf + sign);
    }

} // namespace charconv

class Serializer final
{
public:
    using WriteCallback = std::function<int(const void *, size_t)>;
    using FlushCallback = std::function<int()>;
    using WaitCallback = std::function<void()>;

    static inline constexpr const size_t const_buffer_size_min = 64;
    static inline constexpr const size_t const_fmt_buffer_size_min = 32;

    static inline constexpr const size_t default_buffer_size = 1024;
    static inline constexpr const size_t default_fmt_buffer_size = 64;
    static inline constexpr const int default_floating_point_precision = 3;

    static_assert(default_buffer_size < std::numeric_limits<int>::max(),
        "Default buffer size must be less than INT_MAX");
    static_assert(default_buffer_size >= const_buffer_size_min,
        "Default buffer size must be at least const_buffer_size_min bytes");
    static_assert(const_fmt_buffer_size_min <= default_fmt_buffer_size && default_fmt_buffer_size < default_buffer_size,
        "Default format buffer size must be between const_fmt_buffer_size_min and default_buffer_size (exclusive)");
    static_assert(default_floating_point_precision < default_fmt_buffer_size,
        "Default floating point precision must be less than default_fmt_buffer_size");

    class StringWriter;
    class ForwardWriter;

    class BooleanWriter final
    {
    protected:
        friend class StringWriter;
        friend class ForwardWriter;

        inline constexpr explicit BooleanWriter(Serializer &serializer) noexcept : _serializer(serializer) { }

    public:
        template<typename T>
        static inline constexpr bool is_boolean_v = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, bool>;

        ~BooleanWriter() noexcept { }

        inline constexpr void operator<<(bool value) noexcept
        {
            if (value)
            {
                _serializer.write("true");
            }
            else
            {
                _serializer.write("false");
            }
        }

    private:
        Serializer &_serializer;
    };

    class NumberWriter final
    {
    protected:
        friend class StringWriter;
        friend class ForwardWriter;

        inline constexpr explicit NumberWriter(Serializer &serializer) noexcept : _serializer(serializer) { }

    public:
        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>>
        static inline constexpr bool is_number_v
            = (std::is_integral_v<U> || std::is_floating_point_v<U>) && !std::is_same_v<U, bool>;

        ~NumberWriter() noexcept { }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_integral_v<U>, bool> = true>
        inline constexpr void operator<<(T &&value) noexcept
        {
            if (!_serializer.write(std::forward<T>(value))) [[unlikely]]
            {
                _serializer.write("null");
            }
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_integral_v<U>, bool> = true>
        inline constexpr void operator+=(T &&value) noexcept
        {
            operator<<(std::forward<T>(value));
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
        inline constexpr void operator<<(T &&value) noexcept
        {
            if (std::isnan(value) || std::isinf(value)) [[unlikely]]
            {
                _serializer.write("null");
                return;
            }
            else if (!_serializer.write(std::forward<T>(value), Serializer::default_floating_point_precision))
                [[unlikely]]
            {
                _serializer.write("null");
            }
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
        inline constexpr void operator+=(T &&value) noexcept
        {
            if (std::isnan(value) || std::isinf(value)) [[unlikely]]
            {
                _serializer.write("null");
                return;
            }
            else if (!_serializer.write(std::forward<T>(value))) [[unlikely]]
            {
                _serializer.write("null");
            }
        }

    private:
        Serializer &_serializer;
    };

    class StringWriter final
    {
    protected:
        friend class ForwardWriter;

        inline constexpr explicit StringWriter(Serializer &serializer) noexcept : _serializer(serializer)
        {
            _serializer.write('"');
        }

    public:
        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>>
        static inline constexpr bool is_string_v
            = (std::is_pointer_v<U> && (std::is_same_v<U, const char *> || std::is_same_v<U, char *>))
              || traits::is_bounded_char_array_v<U> || std::is_same_v<U, std::string>
              || std::is_same_v<U, std::string_view>;

        ~StringWriter() noexcept
        {
            _serializer.write('"');
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_pointer_v<U> && (std::is_same_v<U, const char *> || std::is_same_v<U, char *>),
                bool>
            = true>
        inline constexpr StringWriter &operator+=(T &&str) noexcept
        {
            _serializer.write(static_cast<const void *>(str), std::char_traits<char>::length(str));
            return *this;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_bounded_char_array_v<U>, bool> = true>
        inline constexpr StringWriter &operator+=(T &&str) noexcept
        {
            _serializer.write(static_cast<const void *>(str), sizeof(str) - 1);
            return *this;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::string> || std::is_same_v<U, std::string_view>, bool> = true>
        inline constexpr StringWriter &operator+=(T &&str) noexcept
        {
            _serializer.write(static_cast<const void *>(str.data()), str.size());
            return *this;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_pointer_v<U> && (std::is_same_v<U, const char *> || std::is_same_v<U, char *>),
                bool>
            = true>
        inline constexpr StringWriter &operator<<(T &&str) noexcept
        {
            _serializer.write(str, std::char_traits<char>::length(str));
            return *this;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_bounded_char_array_v<U>, bool> = true>
        inline constexpr StringWriter &operator<<(T &&str) noexcept
        {
            _serializer.write(str, sizeof(str) - 1);
            return *this;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::string> || std::is_same_v<U, std::string_view>, bool> = true>
        inline constexpr StringWriter &operator<<(T &&str) noexcept
        {
            _serializer.write(str.data(), str.size());
            return *this;
        }

        template<typename T, std::enable_if_t<BooleanWriter::is_boolean_v<T>, bool> = true>
        inline constexpr StringWriter &operator<<(T &&value) noexcept
        {
            BooleanWriter { _serializer } << std::forward<T>(value);
            return *this;
        }

        template<typename T, std::enable_if_t<NumberWriter::is_number_v<T>, bool> = true>
        inline constexpr StringWriter &operator<<(T &&value) noexcept
        {
            NumberWriter { _serializer } << std::forward<T>(value);
            return *this;
        }

        inline constexpr int write(const void *data, size_t size) noexcept
        {
            if (!data || size == 0) [[unlikely]]
            {
                return 0;
            }
            const int sta = _serializer.state();
            if (sta == 0 && _serializer.write(data, size)) [[likely]]
            {
                return static_cast<int>(size);
            }
            return -sta;
        }

    private:
        Serializer &_serializer;
    };

    class ObjectWriter;

    template<typename F, std::enable_if_t<std::is_same_v<F, ForwardWriter>, bool> = true>
    class ArrayWriter final
    {
    protected:
        friend class ForwardWriter;

        inline constexpr explicit ArrayWriter(Serializer &serializer) noexcept
            : _serializer(serializer), _forward_writer(_serializer), _multiple_items(false)
        {
            _serializer.write('[');
        }

    public:
        ~ArrayWriter() noexcept
        {
            _serializer.write(']');
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, ObjectWriter> || std::is_same_v<U, ArrayWriter<ForwardWriter>>, bool>
            = true>
        inline constexpr U writer() noexcept
        {
            if (_multiple_items) [[likely]]
            {
                _serializer.write(',');
            }
            else [[unlikely]]
            {
                _multiple_items = true;
            }
            return U { _serializer };
        }

        template<typename T>
        inline constexpr ArrayWriter &operator<<(T &&value) noexcept
        {
            if (_multiple_items) [[likely]]
            {
                _serializer.write(',');
            }
            else [[unlikely]]
            {
                _multiple_items = true;
            }
            _forward_writer << std::forward<T>(value);
            return *this;
        }

        template<typename T>
        inline constexpr ArrayWriter &operator+=(T &&value) noexcept
        {
            if (_multiple_items) [[likely]]
            {
                _serializer.write(',');
            }
            else [[unlikely]]
            {
                _multiple_items = true;
            }
            _forward_writer += std::forward<T>(value);
            return *this;
        }

    private:
        Serializer &_serializer;
        F _forward_writer;
        bool _multiple_items;
    };

    class ObjectWriter final
    {
    protected:
        friend class Serializer;
        friend class ForwardWriter;
        friend class ArrayWriter<ForwardWriter>;

        inline explicit ObjectWriter(Serializer &serializer) noexcept : _serializer(serializer), _multiple_items(false)
        {
            _serializer._ref_count.fetch_add(1, std::memory_order_relaxed);
            _serializer.write('{');
        }

    public:
        ~ObjectWriter() noexcept
        {
            _serializer.write('}');
            if (_serializer._ref_count.load(std::memory_order_acquire) == 1) [[unlikely]]
            {
                _serializer.flush();
            }
            _serializer._ref_count.fetch_sub(1, std::memory_order_release);
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_pointer_v<U> && (std::is_same_v<U, const char *> || std::is_same_v<U, char *>),
                bool>
            = true>
        inline ForwardWriter operator[](T &&key) noexcept
        {
            write(static_cast<const void *>(key), std::char_traits<char>::length(key));
            return ForwardWriter { _serializer };
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_bounded_char_array_v<U>, bool> = true>
        inline ForwardWriter operator[](T &&key) noexcept
        {
            write(static_cast<const void *>(&key), sizeof(key) - 1);
            return ForwardWriter { _serializer };
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::string> || std::is_same_v<U, std::string_view>, bool> = true>
        inline ForwardWriter operator[](T &&key) noexcept
        {
            write(static_cast<const void *>(key.data()), key.size());
            return ForwardWriter { _serializer };
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_integral_v<U>, bool> = true>
        inline ForwardWriter operator[](T &&key) noexcept
        {
            write(std::forward<T>(key));
            return ForwardWriter { _serializer };
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_pointer_v<U> && (std::is_same_v<U, const char *> || std::is_same_v<U, char *>),
                bool>
            = true>
        inline ForwardWriter operator()(T &&key) noexcept
        {
            write(key, std::char_traits<char>::length(key));
            return ForwardWriter { _serializer };
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_bounded_char_array_v<U>, bool> = true>
        inline ForwardWriter operator()(T &&key) noexcept
        {
            write(key, sizeof(key) - 1);
            return ForwardWriter { _serializer };
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::string> || std::is_same_v<U, std::string_view>, bool> = true>
        inline ForwardWriter operator()(T &&key) noexcept
        {
            write(key.data(), key.size());
            return ForwardWriter { _serializer };
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_stl_map_container_v<U>, bool> = true>
        inline constexpr ObjectWriter &operator<<(T &&map) noexcept
        {
            for (const auto &[key, value]: map)
            {
                operator()(key) << value;
            }
            return *this;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_stl_map_container_v<U>, bool> = true>
        inline constexpr ObjectWriter &operator+=(T &&map) noexcept
        {
            for (const auto &[key, value]: map)
            {
                operator[](key) += value;
            }
            return *this;
        }

    private:
        template<typename T>
        inline constexpr void write(T &&value, size_t len) noexcept
        {
            if (!len) [[unlikely]]
            {
                return;
            }

            if (_multiple_items) [[likely]]
            {
                _serializer.write(',');
            }
            else [[unlikely]]
            {
                _multiple_items = true;
            }

            _serializer.write('"');
            _serializer.write(std::forward<T>(value), len);
            _serializer.write("\":");
        }

        template<typename T>
        inline constexpr void write(T &&value) noexcept
        {
            if (_multiple_items) [[likely]]
            {
                _serializer.write(',');
            }
            else [[unlikely]]
            {
                _multiple_items = true;
            }

            _serializer.write('"');
            _serializer.write(std::forward<T>(value));
            _serializer.write("\":");
        }

        Serializer &_serializer;
        bool _multiple_items;
    };

    class ForwardWriter final
    {
    protected:
        friend class StringWriter;
        friend class ObjectWriter;
        friend class ArrayWriter<ForwardWriter>;

        inline constexpr explicit ForwardWriter(Serializer &serializer) noexcept : _serializer(serializer) { }

    public:
        ~ForwardWriter() noexcept { }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, StringWriter> || std::is_same_v<U, ObjectWriter>
                                 || std::is_same_v<U, ArrayWriter<ForwardWriter>>,
                bool>
            = true>
        inline constexpr U writer() noexcept
        {
            return U { _serializer };
        }

        inline ObjectWriter operator()() noexcept
        {
            return ObjectWriter { _serializer };
        }

        template<typename T, std::enable_if_t<BooleanWriter::is_boolean_v<T>, bool> = true>
        inline constexpr void operator<<(T &&value) noexcept
        {
            BooleanWriter { _serializer } << std::forward<T>(value);
        }

        template<typename T, std::enable_if_t<BooleanWriter::is_boolean_v<T>, bool> = true>
        inline constexpr void operator+=(T &&value) noexcept
        {
            BooleanWriter { _serializer } << std::forward<T>(value);
        }

        template<typename T, std::enable_if_t<NumberWriter::is_number_v<T>, bool> = true>
        inline constexpr void operator<<(T &&value) noexcept
        {
            NumberWriter { _serializer } << std::forward<T>(value);
        }

        template<typename T, std::enable_if_t<NumberWriter::is_number_v<T>, bool> = true>
        inline constexpr void operator+=(T &&value) noexcept
        {
            NumberWriter { _serializer } += std::forward<T>(value);
        }

        template<typename T, std::enable_if_t<StringWriter::is_string_v<T>, bool> = true>
        inline StringWriter operator<<(T &&value) noexcept
        {
            StringWriter writer { _serializer };
            writer << std::forward<T>(value);
            return writer;
        }

        template<typename T, std::enable_if_t<StringWriter::is_string_v<T>, bool> = true>
        inline StringWriter operator+=(T &&value) noexcept
        {
            StringWriter writer { _serializer };
            writer += std::forward<T>(value);
            return writer;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_variant_v<U>, bool> = true>
        inline constexpr void operator<<(T &&variant) noexcept
        {
            if (variant.valueless_by_exception()) [[unlikely]]
            {
                _serializer.write("null");
                return;
            }
            std::visit([this](const auto &value) constexpr noexcept { *this << value; }, std::forward<T>(variant));
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_variant_v<U>, bool> = true>
        inline constexpr void operator+=(T &&variant) noexcept
        {
            if (variant.valueless_by_exception()) [[unlikely]]
            {
                _serializer.write("null");
                return;
            }
            std::visit([this](const auto &value) constexpr noexcept { *this += value; }, std::forward<T>(variant));
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_stl_container_v<U> && !StringWriter::is_string_v<T>
                                 && !traits::is_stl_map_container_v<U>,
                bool>
            = true>
        inline ArrayWriter<ForwardWriter> operator<<(T &&value) noexcept
        {
            ArrayWriter<ForwardWriter> writer { _serializer };
            for (const auto &item: value)
            {
                writer << item;
            }
            return writer;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_stl_container_v<U> && !StringWriter::is_string_v<T>
                                 && !traits::is_stl_map_container_v<U>,
                bool>
            = true>
        inline ArrayWriter<ForwardWriter> operator+=(T &&value) noexcept
        {
            ArrayWriter<ForwardWriter> writer { _serializer };
            for (const auto &item: value)
            {
                writer += item;
            }
            return writer;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_stl_map_container_v<U>, bool> = true>
        inline ObjectWriter operator<<(T &&map) noexcept
        {
            ObjectWriter writer { _serializer };
            for (const auto &[key, value]: map)
            {
                writer(key) << value;
            }
            return writer;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<traits::is_stl_map_container_v<U>, bool> = true>
        inline ObjectWriter operator+=(T &&map) noexcept
        {
            ObjectWriter writer { _serializer };
            for (const auto &[key, value]: map)
            {
                writer[key] += value;
            }
            return writer;
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::nullptr_t>, bool> = true>
        inline constexpr void operator<<(T &&) noexcept
        {
            _serializer.write("null");
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::nullptr_t>, bool> = true>
        inline constexpr void operator+=(T &&) noexcept
        {
            _serializer.write("null");
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::monostate>, bool> = true>
        inline constexpr void operator<<(T &&) noexcept
        {
            _serializer.write("null");
        }

        template<typename T, typename U = std::remove_cv_t<std::remove_reference_t<T>>,
            std::enable_if_t<std::is_same_v<U, std::monostate>, bool> = true>
        inline constexpr void operator+=(T &&) noexcept
        {
            _serializer.write("null");
        }

    private:
        Serializer &_serializer;
    };

    friend class BooleanWriter;
    friend class NumberWriter;
    friend class StringWriter;
    friend class ArrayWriter<ForwardWriter>;
    friend class ObjectWriter;
    friend class ForwardWriter;

    template<typename T = std::unique_ptr<Serializer>>
    static T create(char prefix, WriteCallback &&write_callback = nullptr, FlushCallback &&flush_callback = nullptr,
        void *buffer = nullptr, size_t size = 0) noexcept

    {
        size = size ? size : default_buffer_size;

        if (size < const_buffer_size_min) [[unlikely]]
        {
            LOG(ERROR, "Buffer size is smaller than %zu bytes (%zu), cannot create serializer", const_buffer_size_min,
                size);
            return {};
        }

        bool internal_buffer = false;
        if (!buffer)
        {
            buffer = static_cast<void *>(new (std::nothrow) unsigned char[size]);
            if (!buffer) [[unlikely]]
            {
                LOG(ERROR, "Failed to allocate buffer of size %zu", size);
                return {};
            }
            internal_buffer = true;
        }

        T ptr = T(new (std::nothrow) Serializer(internal_buffer, static_cast<unsigned char *>(buffer), size, prefix,
            std::move(write_callback), std::move(flush_callback)));
        if (!ptr) [[unlikely]]
        {
            LOG(ERROR, "Failed to create Serializer with buffer size %zu", size);
            if (internal_buffer)
            {
                delete[] static_cast<unsigned char *>(buffer);
            }
            return {};
        }

        return ptr;
    }

    ~Serializer() noexcept
    {
        if (_ref_count.load(std::memory_order_acquire) > 0) [[unlikely]]
        {
            LOG(ERROR, "Serializer destroyed while still in use, flushing remaining data");
            flush();
        }
        if (_internal_buffer && _buffer) [[likely]]
        {
            delete[] _buffer;
        }
        _buffer = nullptr;
    }

    Serializer(const Serializer &) = delete;
    Serializer &operator=(const Serializer &) = delete;

    Serializer(Serializer &&) = delete;
    Serializer &operator=(Serializer &&) = delete;

    ObjectWriter writer(WaitCallback wait_callback = nullptr, WriteCallback &&write_callback = nullptr,
        FlushCallback &&flush_callback = nullptr) noexcept
    {
        const size_t desired = 1;
        while (true)
        {
            size_t expected = 0;
            if (_ref_count.compare_exchange_weak(expected, desired, std::memory_order_acquire,
                    std::memory_order_relaxed))
            {
                break;
            }
            if (wait_callback)
            {
                wait_callback();
            }
        }

        if (write_callback)
        {
            _write_callback = std::move(write_callback);
        }
        if (flush_callback)
        {
            _flush_callback = std::move(flush_callback);
        }

        if (_prefix != '\0')
        {
            write(_prefix);
        }
        auto writer = ObjectWriter(*this);
        _ref_count.fetch_sub(1, std::memory_order_release);

        return writer;
    }

    inline int state() const noexcept
    {
        return _error_code;
    }

protected:
    explicit Serializer(bool internal_buffer, unsigned char *buffer, size_t size, char prefix,
        WriteCallback &&write_callback, FlushCallback &&flush_callback) noexcept
        : _ref_count(0), _internal_buffer(internal_buffer), _buffer_size(size), _buffer_s(buffer),
          _buffer_e(_buffer_s + _buffer_size), _buffer_e_fmt(_buffer_e - default_fmt_buffer_size), _buffer(_buffer_s),
          _error_code(0), _prefix(prefix), _write_callback(std::move(write_callback)),
          _flush_callback(std::move(flush_callback))
    {
        if (!_write_callback) [[unlikely]]
        {
            _write_callback = [](const void *data, size_t size) -> int {
                if (!data || size == 0)
                {
                    return 0;
                }
                return std::fwrite(data, 1, size, stdout);
            };
        }

        if (!_flush_callback) [[unlikely]]
        {
            _flush_callback = []() -> int {
                std::fwrite("\n", 1, 1, stdout);
                const int ret = std::fflush(stdout);
                if (ret != 0) [[unlikely]]
                {
                    return ret;
                }
                return 0;
            };
        }
    }

    constexpr void error(int error_code) noexcept
    {
        if (_error_code == 0)
        {
            _error_code = error_code;
        }
    }

    inline constexpr void sync() noexcept
    {
        const unsigned char *end = _buffer;
        const unsigned char *ptr = _buffer_s;
        while (ptr < end) [[likely]]
        {
            const int ret = _write_callback(ptr, static_cast<size_t>(end - ptr));
            if (ret < 0) [[unlikely]]
            {
                error(-ret);
                break;
            }
            ptr += ret;
        }
        _buffer = _buffer_s;
    }

    inline constexpr void flush() noexcept
    {
        if (_buffer > _buffer_s) [[likely]]
        {
            sync();
        }

        const int ret = _flush_callback();
        if (ret < 0) [[unlikely]]
        {
            error(-ret);
        }
    }

    inline constexpr void write(const char symbol) noexcept
    {
        unsigned char *buffer = _buffer;
        if (buffer >= _buffer_e) [[unlikely]]
        {
            sync();
            buffer = _buffer;
        }
        *buffer = static_cast<unsigned char>(symbol);
        _buffer = buffer + 1;
    }

    template<size_t N, std::enable_if_t<N != 1, bool> = true>
    inline constexpr void write(const char (&symbol)[N]) noexcept
    {
        constexpr const size_t size = N - 1;
        unsigned char *buffer = _buffer;
        unsigned char *buffer_e = buffer + size;
        if (buffer_e > _buffer_e) [[unlikely]]
        {
            sync();
            buffer = _buffer;
            buffer_e = buffer + size;
        }
        std::memcpy(buffer, symbol, size);
        _buffer = buffer_e;
    }

    template<typename T,
        std::enable_if_t<std::is_integral_v<std::remove_cv_t<std::remove_reference_t<T>>>, bool> = true>
    inline bool write(T &&value) noexcept
    {
        if (_buffer > _buffer_e_fmt) [[unlikely]]
        {
            sync();
        }

        char *buffer = reinterpret_cast<char *>(_buffer);
        char* ptr = charconv::itoa(std::forward<T>(value), buffer);
        _buffer = reinterpret_cast<unsigned char *>(ptr);

        return true;
    }

    template<typename T,
        std::enable_if_t<std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<T>>>, bool> = true>
    inline constexpr bool write(T &&value) noexcept
    {
        if (_buffer > _buffer_e_fmt) [[unlikely]]
        {
            sync();
        }

        char *buffer = reinterpret_cast<char *>(_buffer);
        char *buffer_e = reinterpret_cast<char *>(_buffer_e);
        const auto [ptr, ec] = std::to_chars(buffer, buffer_e, value, std::chars_format::fixed);
        if (ec != std::errc()) [[unlikely]]
        {
            error(static_cast<int>(ec));
            return false;
        }
        _buffer = reinterpret_cast<unsigned char *>(ptr);

        return true;
    }

    template<typename T,
        std::enable_if_t<std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<T>>>, bool> = true>
    inline constexpr bool write(T &&value, int precision) noexcept
    {
        if (_buffer > _buffer_e_fmt) [[unlikely]]
        {
            sync();
        }

        char *buffer = reinterpret_cast<char *>(_buffer);
        char *buffer_e = reinterpret_cast<char *>(_buffer_e);
        const auto [ptr, ec] = std::to_chars(buffer, buffer_e, value, std::chars_format::general, precision);
        if (ec != std::errc()) [[unlikely]]
        {
            error(static_cast<int>(ec));
            return false;
        }
        _buffer = reinterpret_cast<unsigned char *>(ptr);

        return true;
    }

    inline constexpr void write(const char *value, size_t len) noexcept
    {
        const char *value_end = value + len;
        const char *value_pos = value;
        while (value < value_end) [[likely]]
        {
            const char c = *value;
            if (static_cast<unsigned char>(c) > 0x1F && c != '"' && c != '\\') [[likely]]
            {
                ++value;
                continue;
            }

            const size_t size = static_cast<size_t>(value - value_pos);
            if (size) [[likely]]
            {
                write(static_cast<const void *>(value_pos), size);
            }
            value_pos = ++value;

            if (c == '"')
            {
                write("\\\"");
            }
            else if (c == '\\')
            {
                write("\\\\");
            }
            else
            {
                switch (c)
                {
                    case '\t':
                        write("\\t");
                        continue;
                    case '\n':
                        write("\\n");
                        continue;
                    case '\f':
                        write("\\f");
                        continue;
                    case '\r':
                        write("\\r");
                        continue;
                    case '\b':
                        write("\\b");
                        continue;
                    default:
                        const char unicode[6] = { '\\', 'u', '0', '0', static_cast<char>(_hex_lookup_tbl[c >> 4]),
                            static_cast<char>(_hex_lookup_tbl[c & 0x0F]) };
                        write(static_cast<const void *>(unicode), sizeof(unicode));
                        continue;
                }
            }
        }
        if (value_pos < value_end) [[likely]]
        {
            write(static_cast<const void *>(value_pos), static_cast<size_t>(value_end - value_pos));
        }
    }

    inline constexpr bool write(const void *data, size_t size) noexcept
    {
        unsigned char *buffer = _buffer;
        unsigned char *buffer_e = buffer + size;
        if (buffer_e <= _buffer_e)
        {
            std::memcpy(buffer, data, size);
            _buffer = buffer_e;
            return true;
        }

        sync();
        if (size <= _buffer_size) [[likely]]
        {
            buffer = _buffer;
            std::memcpy(buffer, data, size);
            _buffer = buffer + size;
            return true;
        }

        const unsigned char *data_ptr = static_cast<const unsigned char *>(data);
        const unsigned char *data_end = data_ptr + size;
        while (data_ptr < data_end) [[likely]]
        {
            const int ret = _write_callback(data_ptr, static_cast<size_t>(data_end - data_ptr));
            if (ret < 0) [[unlikely]]
            {
                error(-ret);
                return false;
            }
            data_ptr += ret;
        }

        return true;
    }

protected:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> _ref_count;
#pragma GCC diagnostic pop

private:
    const bool _internal_buffer;
    const size_t _buffer_size;
    unsigned char *_buffer_s;
    unsigned char *_buffer_e;
    unsigned char *_buffer_e_fmt;
    unsigned char *_buffer;
    int _error_code;
    const char _prefix;
    WriteCallback _write_callback;
    FlushCallback _flush_callback;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"
    alignas(std::hardware_destructive_interference_size) static inline constexpr const
        char _hex_lookup_tbl[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
#pragma GCC diagnostic pop
};

using BooleanWriter = Serializer::BooleanWriter;
using NumberWriter = Serializer::NumberWriter;
using StringWriter = Serializer::StringWriter;
using ObjectWriter = Serializer::ObjectWriter;
using ForwardWriter = Serializer::ForwardWriter;
using ArrayWriter = Serializer::ArrayWriter<ForwardWriter>;

} // namespace core

#endif // SERIALIZER_HPP
