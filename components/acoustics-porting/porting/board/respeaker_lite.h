#pragma once
#ifndef BOARD_RESPEAKER_LITE_H
#define BOARD_RESPEAKER_LITE_H

// Board: Seeed Studio ReSpeaker Lite
// Microphone: I2S XMOS (Slave Mode)
// Sample Rate: 16kHz

#include <driver/gpio.h>
#include <driver/i2s_std.h>

#define BOARD_DEVICE_NAME "ReSpeaker Lite"
// I2S Configuration
#define BOARD_USE_PDM_MODE     0
#define BOARD_USE_I2S_STD_MODE 1

// I2S Role
#define BOARD_I2S_ROLE I2S_ROLE_SLAVE

// I2S Standard Mode Pins
#define BOARD_I2S_BCLK_PIN GPIO_NUM_8
#define BOARD_I2S_WS_PIN   GPIO_NUM_7
#define BOARD_I2S_DIN_PIN  GPIO_NUM_44
#define BOARD_I2S_DOUT_PIN GPIO_NUM_43

// Sample Rate
#define BOARD_RAW_SAMPLE_RATE 16000

// DMA Configuration
#define BOARD_DMA_FRAME_COUNT 800

// GPIO Pins for device control
#define BOARD_GPIO_PINS { 1, 2, 3, 21, 41, 42 }

#endif // BOARD_RESPEAKER_LITE_H
