#pragma once
#ifndef BOARD_DETECTOR_H
#define BOARD_DETECTOR_H

#include <cstdint>

#include <driver/gpio.h>
#include <driver/i2c_master.h>
#include <driver/i2s_pdm.h>
#include <driver/i2s_std.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "core/logger.hpp"

namespace porting {

enum class BoardType { UNKNOWN = 0, XIAO_S3, RESPEAKER_LITE, RESPEAKER_XVF3800 };

struct BoardConfig
{
    BoardType type;
    const char *name;
    bool use_pdm;
    gpio_num_t pdm_clk;
    gpio_num_t pdm_din;
    gpio_num_t i2s_bclk;
    gpio_num_t i2s_ws;
    gpio_num_t i2s_din;
    gpio_num_t i2s_dout;
    i2s_role_t i2s_role;
    uint32_t sample_rate;
    const int *gpio_pins;
    size_t gpio_pins_count;
};

#ifdef DYN_BOARD_TYPE_FROM_I2C_ONCE
#error "DYN_BOARD_TYPE_FROM_I2C is already defined"
#endif
#define DYN_BOARD_TYPE_FROM_I2C_ONCE porting::__DynamicBoardTypeFromI2COnce()

inline BoardType __DynamicBoardTypeFromI2COnce() noexcept
{
    static auto type = []() noexcept -> BoardType {
        LOG(INFO, "Detecting board type via I2C...");

        i2c_master_bus_config_t i2c_cfg = {
            .i2c_port = I2C_NUM_0,
            .sda_io_num = GPIO_NUM_5,
            .scl_io_num = GPIO_NUM_6,
            .clk_source = I2C_CLK_SRC_DEFAULT,
            .glitch_ignore_cnt = 7,
            .intr_priority = 1,
            .trans_queue_depth = 0,
            .flags = { .enable_internal_pullup = true, .allow_pd = 0 },
        };

        i2c_master_bus_handle_t bus_handle;
        if (i2c_new_master_bus(&i2c_cfg, &bus_handle) != ESP_OK)
        {
            LOG(WARNING, "Failed to init I2C, defaulting to XIAO S3");
            return BoardType::XIAO_S3;
        }

        // Check for ReSpeaker Lite (I2C address 0x42)
        if (i2c_master_probe(bus_handle, 0x42, 100) == ESP_OK)
        {
            i2c_del_master_bus(bus_handle);
            LOG(INFO, "Detected: ReSpeaker Lite (I2C 0x42)");
            return BoardType::RESPEAKER_LITE;
        }

        // Check for 0x2C device
        if (i2c_master_probe(bus_handle, 0x2C, 100) == ESP_OK)
        {
            i2c_del_master_bus(bus_handle);
            LOG(INFO, "Detected: ReSpeaker XVF3800 (I2C 0x2C)");
            return BoardType::RESPEAKER_XVF3800;
        }

        i2c_del_master_bus(bus_handle);

        LOG(INFO, "Default: XIAO ESP32-S3");
        return BoardType::XIAO_S3;
    }();

    return type;
}

#ifdef DYN_BOARD_CONFIG_FORM_TYPE
#error "DYN_BOARD_CONFIG_FORM_TYPE is already defined"
#endif
#define DYN_BOARD_CONFIG_FORM_TYPE(type) porting::__DynamicBoardConfigFromType(type)

inline const BoardConfig &__DynamicBoardConfigFromType(BoardType type) noexcept
{
    static const int xiao_s3_gpios[] = { 1, 2, 3, 21, 41, 42 };
    static const int respeaker_lite_gpios[] = { 1, 2, 3, 21, 41, 42 };
    static const int respeaker_xvf3800_gpios[] = { 1, 3, 4, 43, 44 };

    static const BoardConfig configs[] = {
        { BoardType::XIAO_S3, "XIAO ESP32-S3", true, GPIO_NUM_42, GPIO_NUM_41, GPIO_NUM_NC, GPIO_NUM_NC, GPIO_NUM_NC,
            GPIO_NUM_NC, I2S_ROLE_MASTER, 44100, xiao_s3_gpios, 6 },
        { BoardType::RESPEAKER_LITE, "ReSpeaker Lite", false, GPIO_NUM_NC, GPIO_NUM_NC, GPIO_NUM_8, GPIO_NUM_7,
            GPIO_NUM_44, GPIO_NUM_43, I2S_ROLE_SLAVE, 16000, respeaker_lite_gpios, 6 },
        { BoardType::RESPEAKER_XVF3800, "ReSpeaker XVF3800", false, GPIO_NUM_NC, GPIO_NUM_NC, GPIO_NUM_8, GPIO_NUM_7,
            GPIO_NUM_43, GPIO_NUM_44, I2S_ROLE_MASTER, 16000, respeaker_xvf3800_gpios, 5 },
    };

    for (const auto &cfg: configs)
    {
        if (cfg.type == type)
            return cfg;
    }
    return configs[0]; // Default to ESP32-S3
}

} // namespace porting

#endif // BOARD_DETECTOR_H
