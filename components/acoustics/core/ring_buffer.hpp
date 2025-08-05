#pragma once
#ifndef RING_BUFFER_HPP
#define RING_BUFFER_HPP

#include "logger.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace core {

template<typename T,
    std::enable_if_t<std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>, bool> = true>
class RingBuffer final
{
public:
    template<typename P = std::unique_ptr<RingBuffer<T>>>
    [[nodiscard]] static P create(size_t max_capacity, T *buffer = nullptr, size_t size = 0) noexcept
    {
        if (max_capacity < 4) [[unlikely]]
        {
            LOG(ERROR, "RingBuffer max capacity must be at least 4");
            return {};
        }

        if (size && !buffer) [[unlikely]]
        {
            LOG(ERROR, "Buffer cannot be null if size is provided");
            return {};
        }

        if (max_capacity >= std::numeric_limits<size_t>::max() / sizeof(T)) [[unlikely]]
        {
            LOG(ERROR, "RingBuffer max capacity is too large");
            return {};
        }

        if (max_capacity % 2 != 0) [[unlikely]]
        {
            LOG(ERROR, "RingBuffer max capacity must be a multiple of 2");
            return {};
        }

        if (buffer && size < max_capacity * sizeof(T)) [[unlikely]]
        {
            LOG(ERROR, "Provided size (%zu) is less than required size (%zu)", size, max_capacity * sizeof(T));
            return {};
        }

        bool internal_buffer = false;
        if (!buffer)
        {
            buffer = new (std::nothrow) T[max_capacity] {};
            if (!buffer) [[unlikely]]
            {
                LOG(ERROR, "Failed to allocate memory for RingBuffer, size: %zu bytes", max_capacity * sizeof(T));
                return {};
            }
            internal_buffer = true;
        }

        P ptr(new (std::nothrow) RingBuffer<T>(internal_buffer, buffer, max_capacity));
        if (!ptr) [[unlikely]]
        {
            LOG(ERROR, "Failed to create RingBuffer, size: %zu bytes", sizeof(RingBuffer<T>));
            if (internal_buffer)
            {
                delete[] buffer;
            }
            return {};
        }

        return ptr;
    }

    ~RingBuffer() noexcept
    {
        if (_internal_buffer && _buffer)
        {
            delete[] _buffer;
        }
        _buffer = nullptr;
    }

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;

    RingBuffer(RingBuffer &&other) = delete;
    RingBuffer &operator=(RingBuffer &&other) = delete;

    inline size_t size() const noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_relaxed);
        if (head >= tail)
        {
            return head - tail;
        }
        return _max_capacity - (tail - head);
    }

    inline size_t capacity() const noexcept
    {
        return _capacity_mask;
    }

    inline bool empty() const noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_relaxed);
        return head == tail;
    }

    const T *data() const noexcept
    {
        return _buffer;
    }

    template<typename P, std::enable_if_t<std::is_same_v<std::remove_cvref_t<P>, T>, bool> = true>
    inline size_t put(P &&value, bool try_overwrite = true) noexcept
    {
        size_t head = _head.load(std::memory_order_relaxed);
        size_t tail = _tail.load(std::memory_order_acquire);

        T *ptr = _buffer + head;

        const size_t next_head = (head + 1) & _capacity_mask;
        if (next_head == tail) [[unlikely]]
        {
            if (!try_overwrite)
            {
                return 0;
            }

            if (_tail.compare_exchange_strong(tail, (next_head + 1) & _capacity_mask, std::memory_order_release,
                    std::memory_order_relaxed)) [[likely]]
            {
                if (ptr == _lock.load(std::memory_order_relaxed)) [[unlikely]]
                {
                    return 0;
                }
            }
        }

        *ptr = std::forward<P>(value);

        return _head.compare_exchange_strong(head, next_head, std::memory_order_release, std::memory_order_relaxed) ? 1
                                                                                                                    : 0;
    }

    inline size_t get(T &value) noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        size_t tail = _tail.load(std::memory_order_relaxed);
        if (head == tail) [[unlikely]]
        {
            return 0;
        }

        T *ptr = _buffer + tail;

        while (1)
        {
            _lock.store(ptr, std::memory_order_release);

            const size_t current_tail = _tail.load(std::memory_order_relaxed);
            if (current_tail == tail) [[likely]]
            {
                break;
            }
            tail = current_tail;
            ptr = _buffer + tail;
        }

        value = *ptr;

        _tail.compare_exchange_strong(tail, (tail + 1) & _capacity_mask, std::memory_order_release,
            std::memory_order_relaxed);

        _pos = 0;

        _lock.store(nullptr, std::memory_order_release);

        return 1;
    }

    inline size_t write(const T *buffer, size_t count, bool overwrite = true) noexcept
    {
        if (!buffer || !count) [[unlikely]]
        {
            return 0;
        }

        const size_t rb_capacity = capacity();
        if (count > rb_capacity) [[unlikely]]
        {
            buffer += count - rb_capacity;
            count = rb_capacity;
        }

        size_t head = _head.load(std::memory_order_relaxed);
        const size_t tail = _tail.load(std::memory_order_acquire);

        const size_t available = head >= tail ? rb_capacity - head + tail : tail - head - 1;
        const size_t safe_count = std::min(count, available);

        const size_t part_1_start = head;
        const size_t part_1_end = std::min(part_1_start + safe_count, rb_capacity);
        const size_t part_1_count = part_1_end - part_1_start;
        if (part_1_count)
        {
            std::memcpy(_buffer + part_1_start, buffer, part_1_count * sizeof(T));
        }
        const size_t part_2_count = safe_count - part_1_count;
        if (part_2_count)
        {
            std::memcpy(_buffer, buffer + part_1_count, part_2_count * sizeof(T));
        }
        size_t written = safe_count;

        _head.compare_exchange_strong(head, (head + safe_count) & _capacity_mask, std::memory_order_release,
            std::memory_order_relaxed);

        if (overwrite)
        {
            while (written < count)
            {
                written += put(*(buffer + written), true);
            }
        }

        return written;
    }

    inline size_t read(T *buffer, size_t count) noexcept
    {
        if (count == 0) [[unlikely]]
        {
            return 0;
        }

        size_t head = _head.load(std::memory_order_acquire);
        size_t tail = _tail.load(std::memory_order_relaxed);
        if (head == tail) [[unlikely]]
        {
            return 0;
        }

        T *ptr = _buffer + tail;

        while (1)
        {
            _lock.store(ptr, std::memory_order_release);

            const size_t current_tail = _tail.load(std::memory_order_acquire);
            if (current_tail == tail) [[likely]]
            {
                break;
            }
            head = _head.load(std::memory_order_relaxed);
            tail = current_tail;
            ptr = _buffer + tail;
        }

        size_t read = 0;
        size_t next_tail = tail;
        if (buffer)
        {
            while (read < count && next_tail != head)
            {
                buffer[read++] = *ptr;
                next_tail = (next_tail + 1) & _capacity_mask;
                ptr = _buffer + next_tail;
            }
        }
        else
        {
            const size_t size = head >= tail ? head - tail : _max_capacity - tail + head;
            if (count > size) [[unlikely]]
            {
                count = size;
            }
            read = count;
            next_tail = (tail + count) & _capacity_mask;
        }

        size_t expected_tail = tail;
        while (expected_tail != next_tail)
        {
            if (_tail.compare_exchange_strong(expected_tail, next_tail, std::memory_order_release,
                    std::memory_order_relaxed)) [[likely]]
            {
                break;
            }
            tail = (tail + 1) & _capacity_mask;
            expected_tail = tail;
        }

        _pos = 0;

        _lock.store(nullptr, std::memory_order_release);

        return read;
    }

    inline size_t tellg() const noexcept
    {
        return _pos;
    }

    inline bool seekg(size_t pos) const noexcept
    {
        if (pos >= size()) [[unlikely]]
        {
            return false;
        }

        _pos = pos;

        return true;
    }

    inline bool peek(T &value) const noexcept
    {
        size_t head = _head.load(std::memory_order_acquire);
        size_t tail = _tail.load(std::memory_order_relaxed);
        size_t size = head >= tail ? head - tail : _max_capacity - tail + head;
        if (_pos >= size) [[unlikely]]
        {
            return false;
        }

        T *ptr = _buffer + ((tail + _pos) & _capacity_mask);

        while (1)
        {
            _lock.store(ptr, std::memory_order_release);

            const size_t current_tail = _tail.load(std::memory_order_relaxed);
            if (current_tail == tail) [[likely]]
            {
                break;
            }
            tail = current_tail;
            ptr = _buffer + ((tail + _pos) & _capacity_mask);
        }

        value = *ptr;

        _pos += 1;

        _lock.store(nullptr, std::memory_order_release);

        return true;
    }

    void clear() noexcept
    {
        _head.store(0, std::memory_order_release);
        _tail.store(0, std::memory_order_release);

        while (_lock.load(std::memory_order_acquire)) [[unlikely]] { }

        _pos = 0;
    }

protected:
    explicit RingBuffer(bool internal_buffer, T *buffer, size_t max_capacity) noexcept
        : _internal_buffer(internal_buffer), _buffer(buffer), _max_capacity(max_capacity),
          _capacity_mask(_max_capacity - 1), _head(0), _tail(0), _lock(nullptr), _pos(0)
    {
#ifdef __cpp_lib_atomic_is_always_lock_free
        if (!_head.is_always_lock_free || !_tail.is_always_lock_free)
        {
            LOG(WARNING, "RingBuffer head or tail is not lock-free, performance may be affected");
        }

        if (!_lock.is_always_lock_free)
        {
            LOG(WARNING, "RingBuffer lock is not lock-free, performance may be affected");
        }
#endif
    }

private:
    const bool _internal_buffer;
    T *_buffer;
    const size_t _max_capacity;
    const size_t _capacity_mask;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> _head;
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> _tail;
    alignas(std::hardware_destructive_interference_size) std::atomic<void *> mutable _lock;
    alignas(std::hardware_destructive_interference_size) mutable size_t _pos;
#pragma GCC diagnostic pop
};

} // namespace core

#endif // RING_BUFFER_HPP