#pragma once
#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H

#include "board_detector.h"

// Runtime board detection - all configurations included
#define BOARD_USE_RUNTIME_DETECTION 1

// Keep legacy macros for compatibility
#define BOARD_USE_PDM_MODE     0
#define BOARD_USE_I2S_STD_MODE 1

// Default values
#define BOARD_DEVICE_NAME "Runtime Detected"
#define BOARD_RAW_SAMPLE_RATE 16000
#define BOARD_DMA_FRAME_COUNT 800

// GPIO definitions
#define BOARD_PDM_CLK_PIN GPIO_NUM_42
#define BOARD_PDM_DIN_PIN GPIO_NUM_41
#define BOARD_I2S_BCLK_PIN GPIO_NUM_8
#define BOARD_I2S_WS_PIN GPIO_NUM_7
#define BOARD_I2S_DIN_PIN GPIO_NUM_44
#define BOARD_I2S_DOUT_PIN GPIO_NUM_43
#define BOARD_I2S_ROLE I2S_ROLE_MASTER
#define BOARD_GPIO_PINS { 1, 2, 3, 21, 41, 42 }

#endif // BOARD_CONFIG_H
