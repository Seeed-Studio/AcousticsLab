#pragma once
#ifndef BOARD_RESPEAKER_XVF3800_H
#define BOARD_RESPEAKER_XVF3800_H

// Board: Seeed Studio ReSpeaker XVF3800
// Microphone: I2S XMOS (Master Mode)
// Sample Rate: 16kHz

#include <driver/gpio.h>
#include <driver/i2s_std.h>

#define BOARD_DEVICE_NAME "ReSpeaker XVF3800"
// I2S Configuration
#define BOARD_USE_PDM_MODE     0
#define BOARD_USE_I2S_STD_MODE 1

// I2S Role
#define BOARD_I2S_ROLE I2S_ROLE_MASTER

// I2S Standard Mode Pins
#define BOARD_I2S_BCLK_PIN GPIO_NUM_8
#define BOARD_I2S_WS_PIN   GPIO_NUM_7
#define BOARD_I2S_DIN_PIN  GPIO_NUM_43
#define BOARD_I2S_DOUT_PIN GPIO_NUM_44

// Sample Rate
#define BOARD_RAW_SAMPLE_RATE 16000

// DMA Configuration
#define BOARD_DMA_FRAME_COUNT 800

// GPIO Pins for device control
#define BOARD_GPIO_PINS { 1, 3, 4, 43, 44 }

#endif // BOARD_RESPEAKER_XVF3800_H
