#pragma once
#ifndef BOARD_XIAO_S3_H
#define BOARD_XIAO_S3_H

// Board: Seeed Studio XIAO ESP32S3
// Microphone: PDM Microphone
// Sample Rate: 44.1kHz

#include <driver/gpio.h>

#define BOARD_DEVICE_NAME "XIAO ESP32-S3"
// I2S Configuration
#define BOARD_USE_PDM_MODE     1
#define BOARD_USE_I2S_STD_MODE 0

// PDM Microphone Pins
#define BOARD_PDM_CLK_PIN GPIO_NUM_42
#define BOARD_PDM_DIN_PIN GPIO_NUM_41

// Sample Rate
#define BOARD_RAW_SAMPLE_RATE 44100

// DMA Configuration
#define BOARD_DMA_FRAME_COUNT 980

// GPIO Pins for device control
#define BOARD_GPIO_PINS { 1, 2, 3, 21, 41, 42 }

#endif // BOARD_XIAO_S3_H
