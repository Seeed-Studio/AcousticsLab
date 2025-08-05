#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include "hal/engine.hpp"
#include "hal/sensor.hpp"
#include "module/module_dag.hpp"

// Include for SoundClassificationDAG
#include "../../components/acoustics/algorithms/dag/sound_classification_dag.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

std::vector<std::string> generateDynamicLabels(size_t num_classes)
{
    std::vector<std::string> labels;
    labels.reserve(num_classes);

    // Default label mapping
    const std::vector<std::string> default_labels = { "Background Noise", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "unknown" };

    for (size_t i = 0; i < num_classes; ++i)
    {
        if (i < default_labels.size())
        {
            labels.push_back(default_labels[i]);
        }
        else
        {
            labels.push_back("class_" + std::to_string(i));
        }
    }

    return labels;
}

static constexpr size_t ANALYSIS_WINDOW_SAMPLES = 44032;
static constexpr size_t INFERENCE_STRIDE_SAMPLES = 22016;
static constexpr size_t AUDIO_CHUNK_SAMPLES = 4410;

namespace bridge {
extern void __REGISTER_ENGINES__();
extern void __REGISTER_SENSORS__();
extern void __REGISTER_INTERNAL_MODULE_NODE_BUILDER__();
extern void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__();
extern void __REGISTER_MODULE_DAG_BUILDER__();
} // namespace bridge

bool initializeSystem()
{
    std::cout << "=== Initializing Modular Sound Classification System ===" << std::endl;

    std::cout << "Registering system components..." << std::endl;
    bridge::__REGISTER_ENGINES__();
    bridge::__REGISTER_SENSORS__();
    bridge::__REGISTER_INTERNAL_MODULE_NODE_BUILDER__();
    bridge::__REGISTER_EXTERNAL_MODULE_NODE_BUILDER__();
    bridge::__REGISTER_MODULE_DAG_BUILDER__();

    const auto &engines = hal::EngineRegistry::getEngineMap();
    std::cout << "Available engines: " << engines.size() << std::endl;
    for (const auto &[id, engine]: engines)
    {
        std::cout << "  - Engine ID: " << id << ", Name: " << engine->info().name << std::endl;
    }

    std::cout << "All components registered successfully" << std::endl;

    return true;
}

hal::Sensor *initializeSensor()
{
    std::cout << "Initializing audio sensor..." << std::endl;

    auto mic = hal::SensorRegistry::getSensor(2);
    if (!mic)
    {
        std::cout << "ERROR: Sensor with ID 2 not found" << std::endl;
        return nullptr;
    }
    std::cout << "Sensor found: " << mic->info().name << std::endl;

    auto status = mic->init();
    if (!status)
    {
        std::cout << "ERROR: Failed to initialize sensor: " << status.message() << std::endl;
        return nullptr;
    }
    std::cout << "Sensor initialized successfully" << std::endl;

    return mic;
}

std::shared_ptr<module::MDAG> createDAG(const std::string &dag_name, const core::ConfigMap &configs)
{
    std::cout << "Creating DAG: " << dag_name << "..." << std::endl;

    const auto &dag_builders = module::MDAGBuilderRegistry::getDAGBuilderMap();
    std::cout << "Available DAG builders: " << dag_builders.size() << std::endl;
    for (const auto &[name, builder]: dag_builders)
    {
        std::cout << "  - " << name << std::endl;
    }

    const auto &node_builders = module::MNodeBuilderRegistry::getNodeBuilderMap();
    std::cout << "Available node builders: " << node_builders.size() << std::endl;
    for (const auto &[name, builder]: node_builders)
    {
        std::cout << "  - " << name << std::endl;
    }

    auto dag = module::MDAGBuilderRegistry::getDAG(dag_name, configs);
    if (!dag)
    {
        std::cout << "ERROR: Failed to create DAG: " << dag_name << std::endl;
        return nullptr;
    }

    std::cout << "DAG created successfully: " << dag_name << std::endl;

    // Count nodes (forward_list doesn't have size())
    size_t node_count = std::distance(dag->nodes().begin(), dag->nodes().end());
    std::cout << "DAG contains " << node_count << " nodes" << std::endl;

    return dag;
}

void processClassificationResults(core::Tensor *output_tensor, int classification_count,
    std::shared_ptr<module::MDAG> dag)
{
    std::cout << "\n--- Classification Results #" << classification_count << " ---" << std::endl;

    if (!output_tensor)
    {
        std::cout << "ERROR: Output tensor is null" << std::endl;
        return;
    }

    // Determine actual output size by finding the effective data range
    // Since we can't use dynamic_cast (RTTI disabled), we'll analyze the output data
    size_t tensor_capacity = output_tensor->shape().dot();
    size_t num_classes = tensor_capacity; // Default to full capacity

    if (output_tensor->dtype() == core::Tensor::Type::Float32)
    {
        const float *output_data = output_tensor->data<float>();

        // Find the actual output size by looking for the pattern where
        // valid outputs are followed by zeros (unused capacity)
        size_t last_non_zero = 0;
        for (size_t i = 0; i < tensor_capacity; ++i)
        {
            if (output_data[i] != 0.0f)
            {
                last_non_zero = i;
            }
        }

        // If we found non-zero values, the actual size is last_non_zero + 1
        // But we need to be careful about models that legitimately output zeros
        if (last_non_zero < tensor_capacity - 1)
        {
            // Check if there's a clear boundary (many consecutive zeros at the end)
            size_t consecutive_zeros = 0;
            for (size_t i = last_non_zero + 1; i < tensor_capacity; ++i)
            {
                if (output_data[i] == 0.0f)
                {
                    consecutive_zeros++;
                }
                else
                {
                    break;
                }
            }

            // If we have many consecutive zeros at the end, assume that's unused capacity
            if (consecutive_zeros >= 10) // At least 10 consecutive zeros
            {
                num_classes = last_non_zero + 1;
                std::cout << "Detected actual output size: " << num_classes << " classes (capacity: " << tensor_capacity
                          << ")" << std::endl;
            }
            else
            {
                std::cout << "Using full tensor capacity: " << tensor_capacity << " classes" << std::endl;
            }
        }
        else
        {
            std::cout << "Using full tensor capacity: " << tensor_capacity << " classes" << std::endl;
        }
    }
    else
    {
        std::cout << "WARNING: Unsupported tensor type, using full capacity: " << tensor_capacity << " classes"
                  << std::endl;
    }

    // Generate dynamic labels based on actual model output size
    auto labels = generateDynamicLabels(num_classes);

    std::cout << "Output tensor info: capacity=" << output_tensor->shape().dot() << ", actual size=" << num_classes
              << " classes" << std::endl;

    float max_score = -std::numeric_limits<float>::infinity();
    int max_index = 0;

    if (output_tensor->dtype() == core::Tensor::Type::Float32)
    {
        const float *output_data = output_tensor->data<float>();
        if (!output_data)
        {
            std::cout << "ERROR: Failed to get float output data pointer" << std::endl;
            return;
        }

        std::cout << "Classification scores:" << std::endl;
        for (size_t i = 0; i < num_classes; ++i)
        {
            float score = output_data[i];
            std::cout << "  " << labels[i] << ": " << std::fixed << std::setprecision(6) << score << std::endl;

            if (score > max_score)
            {
                max_score = score;
                max_index = i;
            }
        }
    }
    else
    {
        std::cout << "ERROR: Unsupported output tensor data type: " << static_cast<int>(output_tensor->dtype())
                  << std::endl;
        return;
    }

    std::cout << "\n🎯 Predicted class: " << labels[max_index] << " (score: " << std::fixed << std::setprecision(6)
              << max_score << ")" << std::endl;
    std::cout << "--- End Results ---\n" << std::endl;
}

void audioProcessingLoop(hal::Sensor *mic, std::shared_ptr<module::MDAG> dag)
{
    std::cout << "\n=== Starting Streaming Audio Classification ===" << std::endl;
    std::cout << "Processing audio in real-time with sliding window..." << std::endl;
    std::cout << "Window size: " << ANALYSIS_WINDOW_SAMPLES << " samples (1.0s)" << std::endl;
    std::cout << "Stride: " << INFERENCE_STRIDE_SAMPLES << " samples (0.5s)" << std::endl;
    std::cout << "Chunk size: " << AUDIO_CHUNK_SAMPLES << " samples (0.1s)" << std::endl;

    std::unique_ptr<int16_t[]> analysis_window(new int16_t[ANALYSIS_WINDOW_SAMPLES]);
    if (!analysis_window)
    {
        std::cout << "ERROR: Failed to allocate analysis window buffer" << std::endl;
        return;
    }

    memset(analysis_window.get(), 0, ANALYSIS_WINDOW_SAMPLES * sizeof(int16_t));

    auto input_tensor = dag->getInputTensor(0);
    if (!input_tensor)
    {
        std::cout << "ERROR: DAG input tensor not available" << std::endl;
        return;
    }
    if (!input_tensor)
    {
        std::cout << "ERROR: Failed to get DAG input tensor" << std::endl;
        return;
    }

    std::cout << "DAG input tensor: " << (input_tensor->dtype() == core::Tensor::Type::Int16 ? "int16" : "unknown")
              << "[" << input_tensor->shape().dot() << "]" << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    size_t samples_since_last_inference = 0;
    int classification_count = 0;

    std::cout << "\n Starting audio processing loop..." << std::endl;

    while (true)
    {
        auto current_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        float seconds = duration_ms / 1000.0f;

        core::DataFrame<std::unique_ptr<core::Tensor>> audio_chunk;
        auto status = mic->readDataFrame(audio_chunk, AUDIO_CHUNK_SAMPLES);
        if (!status)
        {
            std::cout << "WARNING: Failed to read audio chunk: " << status.message() << std::endl;
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        const int16_t *chunk_data = audio_chunk.data->data<int16_t>();
        if (!chunk_data)
        {
            std::cout << "WARNING: Failed to get chunk data pointer" << std::endl;
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        memmove(analysis_window.get(), analysis_window.get() + AUDIO_CHUNK_SAMPLES,
            (ANALYSIS_WINDOW_SAMPLES - AUDIO_CHUNK_SAMPLES) * sizeof(int16_t));
        memcpy(analysis_window.get() + (ANALYSIS_WINDOW_SAMPLES - AUDIO_CHUNK_SAMPLES), chunk_data,
            AUDIO_CHUNK_SAMPLES * sizeof(int16_t));

        samples_since_last_inference += AUDIO_CHUNK_SAMPLES;

        if (samples_since_last_inference >= INFERENCE_STRIDE_SAMPLES)
        {
            classification_count++;

            std::cout << "\n🔄 Processing classification #" << classification_count << std::endl;
            std::cout << "Time: " << std::fixed << std::setprecision(1) << seconds << "s" << std::endl;

            auto process_start = std::chrono::steady_clock::now();

            int16_t *input_data = input_tensor->data<int16_t>();
            if (!input_data)
            {
                std::cout << "ERROR: Failed to get DAG input tensor data pointer" << std::endl;
                continue;
            }

            memcpy(input_data, analysis_window.get(), ANALYSIS_WINDOW_SAMPLES * sizeof(int16_t));

            auto dag_start = std::chrono::steady_clock::now();
            status = dag->operator()();
            auto dag_end = std::chrono::steady_clock::now();

            if (!status)
            {
                std::cout << "ERROR: DAG execution failed: " << status.message() << std::endl;
                samples_since_last_inference = 0;
                continue;
            }

            auto dag_time = std::chrono::duration_cast<std::chrono::milliseconds>(dag_end - dag_start).count();
            std::cout << "DAG execution time: " << dag_time << "ms" << std::endl;

            auto output_tensor = dag->getOutputTensor(0);
            if (!output_tensor)
            {
                std::cout << "ERROR: DAG output tensor not available" << std::endl;
                continue;
            }
            processClassificationResults(output_tensor.get(), classification_count, dag);

            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - process_start)
                                  .count();
            std::cout << "Total processing time: " << total_time << "ms" << std::endl;

            samples_since_last_inference = 0;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

extern "C" void app_main()
{
    std::cout << "=== Modular Sound Classification Example ===" << std::endl;
    std::cout << "Using Node/DAG architecture with sliding window processing" << std::endl;

    if (!initializeSystem())
    {
        std::cout << "ERROR: System initialization failed" << std::endl;
        return;
    }

    auto mic = initializeSensor();
    if (!mic)
    {
        std::cout << "ERROR: Sensor initialization failed" << std::endl;
        return;
    }

    // Create DAG using generic interface
    core::ConfigMap dag_configs = { { "engine_id", 1 }, { "model_id", 1 }, { "graph_id", 0 } };
    auto dag = createDAG("SoundClassificationDAG", dag_configs);
    if (!dag)
    {
        std::cout << "ERROR: DAG creation failed" << std::endl;
        return;
    }

    std::cout << "\nPerforming initial DAG test..." << std::endl;
    auto test_start = std::chrono::steady_clock::now();
    auto status = dag->operator()();
    auto test_end = std::chrono::steady_clock::now();

    if (!status)
    {
        std::cout << "ERROR: Initial DAG test failed: " << status.message() << std::endl;
        return;
    }

    auto test_time = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start).count();
    std::cout << "Initial DAG test successful: " << test_time << "ms" << std::endl;

    std::cout << "Running warm-up iterations..." << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        auto warmup_start = std::chrono::steady_clock::now();
        status = dag->operator()();
        auto warmup_end = std::chrono::steady_clock::now();

        if (!status)
        {
            std::cout << "WARNING: Warm-up iteration " << i << " failed: " << status.message() << std::endl;
        }
        else
        {
            auto warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(warmup_end - warmup_start).count();
            std::cout << "Warm-up " << i << ": " << warmup_time << "ms" << std::endl;
        }

        vTaskDelay(pdMS_TO_TICKS(5));
    }

    std::cout << "\n✅ System initialization complete!" << std::endl;
    std::cout << "📊 Performance summary:" << std::endl;
    std::cout << "  - DAG execution time: ~" << test_time << "ms" << std::endl;
    std::cout << "  - Audio window: 1.0s (" << ANALYSIS_WINDOW_SAMPLES << " samples)" << std::endl;
    std::cout << "  - Processing stride: 0.5s (" << INFERENCE_STRIDE_SAMPLES << " samples)" << std::endl;
    std::cout << "  - Real-time factor: "
              << (test_time < 500 ? "✅ Real-time capable" : "⚠️ May struggle with real-time") << std::endl;

    std::cout << "\n🚀 Starting main audio processing..." << std::endl;
    audioProcessingLoop(mic, dag);
    std::cout << "Audio processing loop ended unexpectedly" << std::endl;
}
