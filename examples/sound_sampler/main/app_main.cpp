#define EXAMPLE_USE_ADPCM 1
#define EXAMPLE_USE_OPUS  0

#if EXAMPLE_USE_ADPCM

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "core/encoder/adpcm.hpp"
#include "core/encoder/ascii.hpp"
#include "hal/sensor.hpp"

#include <chrono>
#include <iostream>
#include <string_view>

#define REPORT_INTERVAL_SECONDS     0.5
#define PULL_DATA_AVAIL_INTERVAL_MS 100

extern "C" void app_main()
{
    std::cout << "Starting..." << std::endl;

    bridge::__REGISTER_SENSORS__();

    const auto &sensor_map = hal::SensorRegistry::getSensorMap();
    auto it = std::find_if(sensor_map.begin(), sensor_map.end(),
        [](const auto &pair) { return pair.second->info().type == hal::Sensor::Type::Microphone; });
    if (it == sensor_map.end())
    {
        std::cout << "Microphone not found" << std::endl;
        return;
    }
    auto mic = it->second;
    std::cout << "Microphone found: " << mic->info().name << std::endl;
    auto status = mic->init();
    if (!status)
    {
        std::cout << "Failed to initialize microphone: " << status.message() << std::endl;
        return;
    }
    std::cout << "Microphone initialized successfully" << std::endl;

    auto sr = mic->info().configs.at("sr").getValue<int>();
    const size_t sync_elems = static_cast<size_t>(sr * REPORT_INTERVAL_SECONDS);
    const size_t adpcm_buf_sz = sync_elems / 2;
    const size_t base64_buf_sz = adpcm_buf_sz * 2;
    std::cout << "Sample rate: " << sr << " Hz, sync elements: " << sync_elems << " ADPCM buffer size: " << adpcm_buf_sz
              << " bytes, Base64 buffer size: " << base64_buf_sz << " bytes" << std::endl;

    auto adpcm_encoder = core::EncoderADPCMIMA::create(nullptr, adpcm_buf_sz);
    if (!adpcm_encoder || adpcm_encoder->error() != 0)
    {
        std::cout << "Failed to create ADPCM encoder" << std::endl;
        return;
    }

    auto base64_encoder = core::EncoderASCIIBase64::create(nullptr, base64_buf_sz);
    if (!base64_encoder || base64_encoder->error() != 0)
    {
        std::cout << "Failed to create Base64 encoder" << std::endl;
        return;
    }

    auto s = std::chrono::steady_clock::now();
    while (1)
    {
        auto available = mic->dataAvailable();
        if (available < sync_elems)
        {
            vTaskDelay(pdMS_TO_TICKS(PULL_DATA_AVAIL_INTERVAL_MS));
            continue;
        }
        auto duration_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s).count();
        float seconds = duration_ms / 1000.0f;
        std::cout << "Time since start: " << seconds << " seconds, data available: " << available << std::endl;

        auto rs = std::chrono::steady_clock::now();
        auto data_frame = core::DataFrame<std::unique_ptr<core::Tensor>>();
        status = mic->readDataFrame(data_frame, sync_elems);
        auto re = std::chrono::steady_clock::now();
        if (!status)
        {
            std::cout << "Failed to read data frame: " << status.message() << std::endl;
            continue;
        }
        std::cout << "Data frame read successfully in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(re - rs).count() << " ms, "
                  << "data size: " << data_frame.data->size() << " bytes" << std::endl;
        if (data_frame.data->dtype() != core::Tensor::Type::Int16)
        {
            std::cout << "Unexpected data type: " << static_cast<int>(data_frame.data->dtype()) << std::endl;
            break;
        }
        if (sync_elems != data_frame.data->shape().dot())
        {
            std::cout << "Unexpected data size: " << data_frame.data->shape().dot() << std::endl;
            break;
        }

        const void *adpcm_buffer = nullptr;
        size_t adpcm_encoded_bytes = 0;
        const void *base64_buffer = nullptr;
        size_t base64_encoded_bytes = 0;

        auto es = std::chrono::steady_clock::now();
        auto state = adpcm_encoder->state();
        auto adpcm_encoded_sz
            = adpcm_encoder->encode(data_frame.data->data<int16_t>(), sync_elems, [&](const void *data, size_t size) {
                  adpcm_buffer = data;
                  adpcm_encoded_bytes += size;
                  return 0;
              });
        if (adpcm_encoder->error())
        {
            std::cout << "ADPCM encoder error: " << adpcm_encoder->error() << std::endl;
            break;
        }
        if (adpcm_encoded_sz != sync_elems)
        {
            std::cout << "Unexpected ADPCM encoded size: " << adpcm_encoded_sz << std::endl;
            break;
        }

        auto base64_encoded_sz = base64_encoder->encode(static_cast<const uint8_t *>(adpcm_buffer), adpcm_encoded_bytes,
            [&](const void *data, size_t size) {
                base64_buffer = data;
                base64_encoded_bytes += size;
                return 0;
            });
        if (base64_encoder->error())
        {
            std::cout << "Base64 encoder error: " << base64_encoder->error() << std::endl;
            break;
        }
        if (base64_encoded_sz != adpcm_encoded_bytes)
        {
            std::cout << "Unexpected Base64 encoded size: " << base64_encoded_sz << std::endl;
        }
        auto ee = std::chrono::steady_clock::now();

        std::cout << "ADPCM: " << static_cast<int>(state.predictor) << " " << static_cast<int>(state.step_index) << " "
                  << std::string_view(static_cast<const char *>(base64_buffer), base64_encoded_bytes) << std::endl;
        auto se = std::chrono::steady_clock::now();

        std::cout << "Encoding completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(ee - es).count()
                  << " ms, Send completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(se - ee).count()
                  << " ms" << std::endl;
        std::cout << "ADPCM encoded size: " << adpcm_encoded_sz
                  << " items, ADPCM encoded bytes: " << adpcm_encoded_bytes
                  << " bytes, Base64 encoded size: " << base64_encoded_sz
                  << " items, Base64 encoded bytes: " << base64_encoded_bytes << " bytes" << std::endl;
    }

    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

#elif EXAMPLE_USE_OPUS

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "core/encoder/adpcm.hpp"
#include "core/encoder/ascii.hpp"
#include "core/encoder/opus.hpp"
#include "hal/sensor.hpp"

#include <opus.h>

#include <chrono>
#include <iostream>
#include <string_view>

#define REPORT_INTERVAL_SECONDS     0.1
#define PULL_DATA_AVAIL_INTERVAL_MS 10

extern "C" void app_main()
{
    std::cout << "Starting..." << std::endl;

    bridge::__REGISTER_SENSORS__();

    const auto &sensor_map = hal::SensorRegistry::getSensorMap();
    auto it = std::find_if(sensor_map.begin(), sensor_map.end(),
        [](const auto &pair) { return pair.second->info().type == hal::Sensor::Type::Microphone; });
    if (it == sensor_map.end())
    {
        std::cout << "Microphone not found" << std::endl;
        return;
    }
    auto mic = it->second;
    std::cout << "Microphone found: " << mic->info().name << std::endl;
    auto status = mic->init();
    if (!status)
    {
        std::cout << "Failed to initialize microphone: " << status.message() << std::endl;
        return;
    }
    std::cout << "Microphone initialized successfully" << std::endl;

    auto sr = mic->info().configs.at("sr").getValue<int>();
    const size_t sync_elems = static_cast<size_t>(sr * REPORT_INTERVAL_SECONDS);

    auto base64_encoder = core::EncoderASCIIBase64::create(nullptr, 40 * 1024);
    if (!base64_encoder || base64_encoder->error() != 0)
    {
        std::cout << "Failed to create Base64 encoder" << std::endl;
        return;
    }

    auto opus_encoder = core::EncoderLIBOPUS::create(nullptr, 4096, sr);
    if (!opus_encoder || opus_encoder->error() != 0)
    {
        std::cout << "Failed to create OPUS encoder" << std::endl;
        return;
    }

    auto s = std::chrono::steady_clock::now();
    while (1)
    {
        auto available = mic->dataAvailable();
        if (available < sync_elems)
        {
            vTaskDelay(pdMS_TO_TICKS(PULL_DATA_AVAIL_INTERVAL_MS));
            continue;
        }
        auto duration_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s).count();
        float seconds = duration_ms / 1000.0f;
        std::cout << "Time since start: " << seconds << " seconds, data available: " << available << std::endl;

        auto rs = std::chrono::steady_clock::now();
        auto data_frame = core::DataFrame<std::unique_ptr<core::Tensor>>();
        status = mic->readDataFrame(data_frame, sync_elems);
        auto re = std::chrono::steady_clock::now();
        if (!status)
        {
            std::cout << "Failed to read data frame: " << status.message() << std::endl;
            continue;
        }
        std::cout << "Data frame read successfully in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(re - rs).count() << " ms, "
                  << "data size: " << data_frame.data->size() << " bytes" << std::endl;
        if (data_frame.data->dtype() != core::Tensor::Type::Int16)
        {
            std::cout << "Unexpected data type: " << static_cast<int>(data_frame.data->dtype()) << std::endl;
            break;
        }
        if (sync_elems != data_frame.data->shape().dot())
        {
            std::cout << "Unexpected data size: " << data_frame.data->shape().dot() << std::endl;
            break;
        }

        size_t opus_encoded_bytes = 0;
        size_t base64_encoded_bytes = 0;
        auto es = std::chrono::steady_clock::now();
        {
            auto opus_encoded_sz = opus_encoder->encode(data_frame.data->data<int16_t>(), sync_elems,
                [&](const void *data, size_t size) {
                    opus_encoded_bytes += size;
                    base64_encoder->encode(static_cast<const uint8_t *>(data), size,
                        [&](const void *bdata, size_t bsize) {
                            std::cout << "OPUS: " << std::string_view(static_cast<const char *>(bdata), bsize)
                                      << std::endl;
                            base64_encoded_bytes += bsize;
                            return 0;
                        });
                    if (base64_encoder->error() != 0)
                    {
                        std::cout << "Base64 encoder error: " << base64_encoder->error() << std::endl;
                    }
                    return 0;
                });
            if (opus_encoder->error() != 0)
            {
                std::cout << "OPUS encoder error: " << opus_encoder->error() << std::endl;
            }

            std::cout << "OPUS encoded " << opus_encoded_sz << "/" << sync_elems << " items, "
                      << "bytes: " << opus_encoded_bytes << std::endl;
            std::cout << "Base64 encoded bytes: " << base64_encoded_bytes << std::endl;
        }
        auto ee = std::chrono::steady_clock::now();

        std::cout << "Encoding completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(ee - es).count()
                  << " ms" << std::endl;

        vTaskDelay(pdMS_TO_TICKS(10));
    }

    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

#endif
