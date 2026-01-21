#include "board/board_config.h"
#include "board/board_detector.h"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "hal/device.hpp"

#include <bd/lfs_flashbd.h>
#include <lfs.h>

#include <driver/gpio.h>
#include <esp_efuse.h>
#include <esp_efuse_chip.h>
#include <esp_efuse_table.h>
#include <esp_heap_caps.h>
#include <esp_partition.h>
#include <esp_system.h>
#include <hal/efuse_hal.h>
#include <spi_flash_mmap.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <string_view>

namespace porting {

static const char *getDeviceID() noexcept
{
    char id_full[16];
    int ret = esp_efuse_read_field_blob(ESP_EFUSE_OPTIONAL_UNIQUE_ID, id_full, 16u << 3);
    if (ret != ESP_OK)
    {
        return "";
    }

    static char id_str[sizeof(id_full) * 2 + 1] = { 0 };
    for (size_t i = 0; i < sizeof(id_full); ++i)
    {
        std::sprintf(&id_str[i * 2], "%02x", static_cast<unsigned char>(id_full[i]));
    }
    return id_str;
}

static size_t getFreeMemorySize() noexcept
{
    multi_heap_info_t heap_info;
    heap_caps_get_info(&heap_info, MALLOC_CAP_8BIT);
    return heap_info.total_free_bytes;
}

static constexpr const char DEVICE_MODEL[] = "ESP32-S3";
static constexpr const char DEVICE_VERSION[] = "1.0.0";
static constexpr const char DEVICE_NAME[] = BOARD_DEVICE_NAME;
static constexpr const size_t DEVICE_MEMORY_SIZE = 8 * 1024 * 1024;
static constexpr const size_t DEVICE_NAME_LENGTH_MAX = 64;

static constexpr const char DEFAULT_DEVICE_NAME_PATH[] = ".device_name";
static constexpr const char DEFAULT_BOOT_COUNT_PATH[] = ".boot_count";

class DeviceESP32S3 final: public hal::Device
{
public:
    DeviceESP32S3() noexcept
        : Device(Info(getDeviceID(), DEVICE_MODEL, DEVICE_VERSION, DEVICE_MEMORY_SIZE, DEVICE_NAME, 0,
              std::chrono::system_clock::now(), getFreeMemorySize())),
          _storage_lfs_mutex(), _storage_lfs_flashbd_config(), _storage_lfs_flashbd(), _storage_lfs_config(),
          _storage_lfs()
    {
        _storage_lfs_flashbd.flash_addr = nullptr;
    }

    core::Status init() noexcept override
    {
        LOG(INFO, "Initializing ESP32-S3 device");
        if (initialized())
        {
            LOG(WARNING, "Device already initialized, skipping re-initialization");
            return STATUS_OK();
        }

        auto board_type = porting::detectBoard();
        _board_config = porting::getBoardConfig(board_type);
        _info.name = _board_config.name;
        for (size_t i = 0; i < _board_config.gpio_pins_count; ++i)
        {
            int pin = _board_config.gpio_pins[i];
            gpio_set_direction(static_cast<gpio_num_t>(pin), GPIO_MODE_OUTPUT);
            gpio_set_pull_mode(static_cast<gpio_num_t>(pin), GPIO_FLOATING);
        }


        {
            auto status = initLittleFS();
            if (!status)
            {
                return status;
            }

            status = existsInLittleFS(DEFAULT_DEVICE_NAME_PATH);
            if (status)
            {
                size_t device_name_size = 0;
                status = loadFromLittleFS(DEFAULT_DEVICE_NAME_PATH, nullptr, device_name_size);
                if (status && device_name_size > 0 && device_name_size <= DEVICE_NAME_LENGTH_MAX)
                {
                    std::string device_name(device_name_size, '\0');
                    status = loadFromLittleFS(DEFAULT_DEVICE_NAME_PATH, device_name.data(), device_name_size);
                    if (status)
                    {
                        _info.name = std::move(device_name);
                        LOG(INFO, "Loaded device name from LittleFS: '%s'", _info.name.data());
                    }
                    else
                    {
                        LOG(ERROR, "Failed to load device name from LittleFS: %s", status.message().c_str());
                    }
                }
            }

            size_t boot_count_size = sizeof(_info.boot_count);
            status = loadFromLittleFS(DEFAULT_BOOT_COUNT_PATH, &_info.boot_count, boot_count_size);
            if (status && boot_count_size == sizeof(_info.boot_count))
            {
                _info.boot_count += 1;
            }
            else
            {
                _info.boot_count = 1;
                LOG(WARNING, "Boot count not found or size mismatch, initializing to 1");
            }
            status = storeToLittleFS(DEFAULT_BOOT_COUNT_PATH, &_info.boot_count, sizeof(_info.boot_count));
            if (!status)
            {
                LOG(ERROR, "Failed to initialize boot count: %s", status.message().c_str());
            }
        }

        _info.status = Status::Ready;

        return STATUS_OK();
    }

    core::Status deinit() noexcept override
    {
        LOG(INFO, "Deinitializing ESP32-S3 device");

        if (!initialized())
        {
            LOG(WARNING, "Device not initialized or already deinitialized, skipping");
            return STATUS_OK();
        }

        auto status = deinitLittleFS();
        if (!status)
        {
            return status;
        }

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    inline uint32_t timestamp() const noexcept override
    {
        return xTaskGetTickCount() * portTICK_PERIOD_MS;
    }

    inline void sleep(size_t duration_ms) const noexcept override
    {
        vTaskDelay(pdMS_TO_TICKS(duration_ms));
    }

    inline void reset() const noexcept override
    {
        LOG(INFO, "Resetting ESP32-S3 device");
        esp_restart();
    }

    inline int gpio(hal::Device::GPIOOpType op, int pin, int value = 0) noexcept override
    {
        switch (op)
        {
            case hal::Device::GPIOOpType::Config:
                return -ENOTSUP;
            case hal::Device::GPIOOpType::Write:
                for (size_t i = 0; i < _board_config.gpio_pins_count; ++i)
                {
                    if (pin == _board_config.gpio_pins[i])
                    {
                        return gpio_set_level(static_cast<gpio_num_t>(pin), value ? 1 : 0) == ESP_OK ? 0 : -EIO;
                    }
                }
                return -EINVAL;
            case hal::Device::GPIOOpType::Read:
                return -ENOTSUP;
            default:
                return -ENOTSUP;
        }
    }

    core::Status store(StorageType where, std::string path, const void *data, size_t size) noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return STATUS(ENODEV, "Device not initialized");
        }

        switch (where)
        {
            case StorageType::Internal:
                return storeToLittleFS(path, data, size);
            default:
                break;
        }
        return STATUS(ENOTSUP, "Unsupported storage type for store operation");
    }

    core::Status load(StorageType where, std::string path, void *data, size_t &size) noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return STATUS(ENODEV, "Device not initialized");
        }

        switch (where)
        {
            case StorageType::Internal:
                return loadFromLittleFS(path, data, size);
            default:
                break;
        }
        return STATUS(ENOTSUP, "Unsupported storage type for load operation");
    }

    core::Status exists(StorageType where, std::string path) noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return STATUS(ENODEV, "Device not initialized");
        }

        switch (where)
        {
            case StorageType::Internal:
                return existsInLittleFS(path);
            default:
                break;
        }
        return STATUS(ENOTSUP, "Unsupported storage type for exists operation");
    }

    core::Status remove(StorageType where, std::string path) noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return STATUS(ENODEV, "Device not initialized");
        }

        switch (where)
        {
            case StorageType::Internal:
                return removeFromLittleFS(path);
            default:
                break;
        }
        return STATUS(ENOTSUP, "Unsupported storage type for remove operation");
    }

    core::Status erase(StorageType where) noexcept override
    {
        if (!initialized()) [[unlikely]]
        {
            return STATUS(ENODEV, "Device not initialized");
        }

        switch (where)
        {
            case StorageType::Internal:
                return eraseLittleFS();
            default:
                break;
        }
        return STATUS(ENOTSUP, "Unsupported storage type for erase operation");
    }

private:
    void syncInfo(Info &info) noexcept override
    {
        _info.free_memory = getFreeMemorySize();
    }

    bool syncDeviceName(const std::string &new_name) noexcept override
    {
        if (new_name.empty())
        {
            LOG(ERROR, "Device name cannot be empty");
            return false;
        }
        if (new_name.size() > DEVICE_NAME_LENGTH_MAX)
        {
            LOG(ERROR, "Device name cannot exceed %zu characters", DEVICE_NAME_LENGTH_MAX);
            return false;
        }
        if (_info.name == new_name)
        {
            LOG(INFO, "Device name is already set to '%s', no change needed", new_name.c_str());
            return true;
        }

        auto status = storeToLittleFS(DEFAULT_DEVICE_NAME_PATH, new_name.data(), new_name.size());
        if (!status)
        {
            LOG(ERROR, "Failed to store device name: %s", status.message().c_str());
            return false;
        }

        return true;
    }

    core::Status initLittleFS() noexcept
    {
        const auto guard = std::lock_guard<std::mutex>(_storage_lfs_mutex);

        if (_storage_lfs_flashbd.flash_addr != nullptr)
        {
            LOG(WARNING, "LittleFS already initialized, skipping re-initialization");
            return STATUS_OK();
        }

        if (default_littlefs_partition)
        {
            LOG(WARNING, "Default LittleFS partition already set, skipping search");
        }
        else
        {
            default_littlefs_partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                ESP_PARTITION_SUBTYPE_DATA_UNDEFINED, DEFAULT_LFS_PARTITION_NAME);
            if (!default_littlefs_partition)
            {
                LOG(ERROR, "Default LittleFS partition not found");
                return STATUS(ENOENT, "Default LittleFS partition not found");
            }
        }
        LOG(INFO, "Using LittleFS partition: %s at address 0x%p, size %lu bytes", default_littlefs_partition->label,
            reinterpret_cast<void *>(default_littlefs_partition->address), default_littlefs_partition->size);

        _storage_lfs_flashbd_config = {
            .read_size = 16,
            .prog_size = 16,
            .erase_size = 4096,
            .erase_count = default_littlefs_partition->size / 4096,
            .flash_addr = reinterpret_cast<void *>(default_littlefs_partition->address),
        };

        _storage_lfs_config = {
            .context = &_storage_lfs_flashbd,
            .read = lfs_flashbd_read,
            .prog = lfs_flashbd_prog,
            .erase = lfs_flashbd_erase,
            .sync = lfs_flashbd_sync,
            .read_size = _storage_lfs_flashbd_config.read_size,
            .prog_size = _storage_lfs_flashbd_config.prog_size,
            .block_size = _storage_lfs_flashbd_config.erase_size,
            .block_count = _storage_lfs_flashbd_config.erase_count,
            .block_cycles = 1000,
            .cache_size = 16,
            .lookahead_size = 16,
            .compact_thresh = 0,
            .read_buffer = nullptr,
            .prog_buffer = nullptr,
            .lookahead_buffer = nullptr,
            .name_max = 255,
            .file_max = LFS_FILE_MAX,
            .attr_max = LFS_ATTR_MAX,
            .metadata_max = 0,
            .inline_max = 0,
        };

        int ret = lfs_flashbd_create(&_storage_lfs_config, &_storage_lfs_flashbd_config);
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to create LittleFS flash block device: %d", ret);
            return STATUS(ENODEV, "Failed to create LittleFS flash block device");
        }

        ret = lfs_mount(&_storage_lfs, &_storage_lfs_config);
        if (ret != LFS_ERR_OK)
        {
            LOG(WARNING, "LittleFS mount failed with error code: %d", ret);
            ret = lfs_format(&_storage_lfs, &_storage_lfs_config);
            if (ret != LFS_ERR_OK)
            {
                LOG(ERROR, "Failed to format LittleFS: %d", ret);
                lfs_flashbd_destroy(&_storage_lfs_config);
                return STATUS(EFAULT, "Failed to format LittleFS");
            }
            ret = lfs_mount(&_storage_lfs, &_storage_lfs_config);
        }
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to mount LittleFS: %d", ret);
            lfs_flashbd_destroy(&_storage_lfs_config);
            return STATUS(ENOENT, "Failed to mount LittleFS");
        }

        LOG(INFO, "LittleFS initialized successfully");

        return STATUS_OK();
    }

    core::Status deinitLittleFS() noexcept
    {
        const auto guard = std::lock_guard<std::mutex>(_storage_lfs_mutex);

        if (_storage_lfs_flashbd.flash_addr == nullptr)
        {
            LOG(WARNING, "LittleFS not initialized, skipping de-initialization");
            return STATUS_OK();
        }

        int ret = lfs_unmount(&_storage_lfs);
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to unmount LittleFS: %d", ret);
            return STATUS(EBUSY, "Failed to unmount LittleFS");
        }

        ret = lfs_flashbd_destroy(&_storage_lfs_config);
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to destroy LittleFS flash block device: %d", ret);
            return STATUS(EFAULT, "Failed to destroy LittleFS flash block device");
        }

        _storage_lfs_flashbd.flash_addr = nullptr;

        LOG(INFO, "LittleFS de-initialized successfully");

        return STATUS_OK();
    }

    static std::list<std::string> pathToParts(const std::string &path) noexcept
    {
        std::list<std::string> parts;
        size_t pos = 0;
        while (pos < path.size())
        {
            size_t next_pos = path.find('/', pos);
            if (next_pos == std::string::npos)
            {
                next_pos = path.size();
            }
            if (next_pos > pos)
            {
                parts.emplace_back(path.substr(pos, next_pos - pos));
            }
            pos = next_pos + 1;
        }
        return parts;
    }

    core ::Status storeToLittleFS(const std::string &path, const void *data, size_t size) noexcept
    {
        const auto guard = std::lock_guard<std::mutex>(_storage_lfs_mutex);

        auto parts = pathToParts(path);
        if (parts.empty())
        {
            LOG(ERROR, "Invalid path: '%s'", path.c_str());
            return STATUS(EINVAL, "Invalid path");
        }

        std::string safe_path;
        for (auto it = parts.begin(); it != std::prev(parts.end()); ++it)
        {
            safe_path.push_back('/');
            safe_path += std::move(*it);
            int ret = lfs_mkdir(&_storage_lfs, safe_path.c_str());
            if (ret != LFS_ERR_OK && ret != LFS_ERR_EXIST) [[unlikely]]
            {
                LOG(ERROR, "Failed to create directory '%s': %d", safe_path.c_str(), ret);
                return STATUS(ENOENT, "Failed to create directory");
            }
        }
        safe_path.push_back('/');
        safe_path += std::move(parts.back());
        parts.clear();

        lfs_file_t file;
        int ret = lfs_file_open(&_storage_lfs, &file, safe_path.c_str(), LFS_O_RDWR | LFS_O_CREAT | LFS_O_TRUNC);
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to open file '%s': %d", safe_path.c_str(), ret);
            return STATUS(ENOENT, "Failed to open file");
        }

        lfs_ssize_t len = lfs_file_write(&_storage_lfs, &file, data, size);
        if (len < 0)
        {
            LOG(ERROR, "Failed to write to file '%s': %ld", safe_path.c_str(), len);
            lfs_file_close(&_storage_lfs, &file);
            return STATUS(EIO, "Failed to write to file");
        }
        if (static_cast<size_t>(len) != size)
        {
            LOG(ERROR, "Partial write to file '%s': expected %zu bytes, wrote %ld bytes", safe_path.c_str(), size, len);
            lfs_file_close(&_storage_lfs, &file);
            return STATUS(EIO, "Partial write to file");
        }

        ret = lfs_file_close(&_storage_lfs, &file);
        if (ret != LFS_ERR_OK) [[unlikely]]
        {
            LOG(ERROR, "Failed to close file '%s': %d", safe_path.c_str(), ret);
            return STATUS(EIO, "Failed to close file");
        }

        return STATUS_OK();
    }

    core::Status loadFromLittleFS(const std::string &path, void *data, size_t &size) noexcept
    {
        const auto guard = std::lock_guard<std::mutex>(_storage_lfs_mutex);

        if (!data || size == 0)
        {
            lfs_info info;
            int ret = lfs_stat(&_storage_lfs, path.c_str(), &info);
            if (ret != LFS_ERR_OK)
            {
                LOG(ERROR, "Failed to stat file '%s': %d", path.c_str(), ret);
                return STATUS(ENOENT, "Failed to stat file");
            }
            if (info.type != LFS_TYPE_REG)
            {
                LOG(ERROR, "Path '%s' is not a regular file", path.c_str());
                return STATUS(EISDIR, "Path is not a regular file");
            }
            size = info.size;
            return STATUS_OK();
        }

        lfs_file_t file;
        int ret = lfs_file_open(&_storage_lfs, &file, path.c_str(), LFS_O_RDONLY);
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to open file '%s': %d", path.c_str(), ret);
            return STATUS(ENOENT, "Failed to open file");
        }

        lfs_ssize_t len = lfs_file_read(&_storage_lfs, &file, data, size);
        if (len < 0)
        {
            LOG(ERROR, "Failed to read from file '%s': %ld", path.c_str(), len);
            lfs_file_close(&_storage_lfs, &file);
            return STATUS(EIO, "Failed to read from file");
        }

        if (static_cast<size_t>(len) != size)
        {
            LOG(ERROR, "Partial read from file '%s': expected %zu bytes, read %ld bytes", path.c_str(), size, len);
            lfs_file_close(&_storage_lfs, &file);
            return STATUS(EIO, "Partial read from file");
        }

        ret = lfs_file_close(&_storage_lfs, &file);
        if (ret != LFS_ERR_OK) [[unlikely]]
        {
            LOG(ERROR, "Failed to close file '%s': %d", path.c_str(), ret);
            return STATUS(EIO, "Failed to close file");
        }

        return STATUS_OK();
    }

    core::Status existsInLittleFS(const std::string &path) noexcept
    {
        const auto guard = std::lock_guard<std::mutex>(_storage_lfs_mutex);

        lfs_info info;
        int ret = lfs_stat(&_storage_lfs, path.c_str(), &info);
        if (ret == LFS_ERR_OK)
        {
            return STATUS_OK();
        }
        else if (ret == LFS_ERR_NOENT)
        {
            return STATUS(ENOENT, "File does not exist");
        }
        LOG(ERROR, "Failed to stat file '%s': %d", path.c_str(), ret);

        return STATUS(EIO, "Failed to check file existence");
    }

    core::Status removeFromLittleFS(const std::string &path) noexcept
    {
        const auto guard = std::lock_guard<std::mutex>(_storage_lfs_mutex);

        int ret = lfs_remove(&_storage_lfs, path.c_str());
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to remove file '%s': %d", path.c_str(), ret);
            return STATUS(EINVAL, "Failed to remove file");
        }
        return STATUS_OK();
    }

    core::Status eraseLittleFS() noexcept
    {
        const auto guard = std::lock_guard<std::mutex>(_storage_lfs_mutex);

        int ret = lfs_remove(&_storage_lfs, "/");
        if (ret != LFS_ERR_OK)
        {
            LOG(ERROR, "Failed to erase LittleFS: %d", ret);
            return STATUS(EIO, "Failed to erase LittleFS");
        }
        return STATUS_OK();
    }

    std::mutex _storage_lfs_mutex;
    lfs_flashbd_config _storage_lfs_flashbd_config;
    lfs_flashbd_t _storage_lfs_flashbd;
    lfs_config _storage_lfs_config;
    lfs_t _storage_lfs;
    porting::BoardConfig _board_config;
};

} // namespace porting

namespace bridge {

void __REGISTER_DEVICE__()
{
    [[maybe_unused]] static porting::DeviceESP32S3 device_esp32s3;
}

} // namespace bridge
