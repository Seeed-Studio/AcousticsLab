#! /usr/bin/env python3

# pylint: disable=missing-module-docstring, too-many-lines
import argparse
import logging
import math
import os
import sys
import threading
import time
from collections import defaultdict

import gradio as gr
import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from client import (  # pylint: disable=import-error,wrong-import-position # noqa: E402
    Client,
    ResponseType,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

CLIENT = None
SENSOR_CACHE = {
    "data": defaultdict(lambda: np.array([])),
    "limit": 500,
}
CONFIGS = {
    "anomaly_threshold": {
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.001,
    },
    "window_size": {
        "min": 32,
        "max": 768,
        "default": 192,
        "step": 1,
    },
    "chunk_size": {
        "min": 32,
        "max": 256,
        "default": 64,
        "step": 1,
    },
    "number_of_chunks": {
        "min": 10,
        "max": 100,
        "default": 20,
        "step": 1,
    },
    "minimal_n_chunks": {
        "min": 1,
        "max": 20,
        "default": 5,
        "step": 1,
    },
    "slide_steps": {
        "min": 1,
        "max": 64,
        "default": 1,
        "step": 1,
    },
    "chunk_extract_after": {
        "min": 0,
        "max": 512,
        "default": 64,
        "step": 1,
    },
    "event_report_interval": {
        "min": -1,
        "max": 1000,
        "default": 100,
        "step": 1,
    },
    "gravity_alpha": {
        "min": 1.0,
        "max": 100.0,
        "default": 9.78762,
        "step": 0.001,
    },
    "gravity_beta": {
        "min": -100,
        "max": 100,
        "default": 0.0,
        "step": 0.001,
    },
    "abnormal_output_gpio": {
        "min": -1,
        "max": 128,
        "default": -1,
        "step": 1,
    },
    "abnormal_output_gpio_value": {
        "min": 0,
        "max": 1,
        "default": 1,
        "step": 1,
    },
}
for _k, _v in CONFIGS.items():
    if _v["min"] > _v["max"]:
        raise ValueError(f"Invalid configuration for {_k}: min > max")
    if _v["default"] < _v["min"] or _v["default"] > _v["max"]:
        raise ValueError(
            f"Invalid default value for {_k}: {_v['default']} not in [{_v['min']}, {_v['max']}]"
        )
    if "step" not in _v:
        _v["step"] = 1
CONFIG_VALUES = {_k: _v["default"] for _k, _v in CONFIGS.items()}
CONFIG_RELATIVE_CONSTRAINTS = [
    {
        "related": ["chunk_extract_after", "window_size"],
        "expr": lambda x: x["chunk_extract_after"] <= x["window_size"],
        "msg": "'Chunk Extract After' <= 'Window Size'",
    },
    {
        "related": ["chunk_size", "window_size"],
        "expr": lambda x: x["chunk_size"] <= x["window_size"],
        "msg": "'Chunk Size' <= 'Window Size'",
    },
    {
        "related": [
            "chunk_extract_after",
            "number_of_chunks",
            "window_size",
            "chunk_size",
        ],
        "expr": lambda x: x["chunk_extract_after"] + x["number_of_chunks"]
        <= x["window_size"] - x["chunk_size"],
        "msg": "'Chunk Extract After' + 'Number of Chunks' <= 'Window Size' - 'Chunk Size'",
    },
    {
        "related": ["slide_steps", "window_size"],
        "expr": lambda x: x["slide_steps"] <= x["window_size"],
        "msg": "'Slide Steps' <= 'Window Size'",
    },
    {
        "related": ["number_of_chunks", "window_size", "slide_steps"],
        "expr": lambda x: x["number_of_chunks"]
        <= math.floor(x["window_size"] / x["slide_steps"]),
        "msg": "'Number of Chunks' <= 'Window Size' / 'Slide Steps'",
    },
]
CONFIG_RELATIVE_CONSTRAINTS = sorted(
    CONFIG_RELATIVE_CONSTRAINTS, key=lambda x: len(x["related"])
)
CONFIGURABLE_KEYS = [
    "anomaly_threshold",
    "window_size",
    "chunk_size",
    "number_of_chunks",
    "minimal_n_chunks",
    "slide_steps",
    "chunk_extract_after",
    "event_report_interval",
    "gravity_alpha",
    "gravity_beta",
    "abnormal_output_gpio",
    "abnormal_output_gpio_value",
]
CONFIGURABLE_KEYS_STATIC = [
    "window_size",
    "chunk_size",
    "number_of_chunks",
    "minimal_n_chunks",
    "slide_steps",
    "chunk_extract_after",
]
CONFIGURABLE_KEYS_DYNAMIC = [
    "anomaly_threshold",
    "gravity_alpha",
    "gravity_beta",
    "event_report_interval",
]
CONFIGURABLE_GPIOS = {
    "Disabled": {
        "value": -1,
        "choices": {
            "None": 0,
        },
    },
    "LED": {
        "value": 21,
        "choices": {
            "On": 0,
            "Off": 1,
        },
    },
    "GPIO 1": {
        "value": 1,
        "choices": {
            "High": 1,
            "Low": 0,
        },
    },
    "GPIO 2": {
        "value": 2,
        "choices": {
            "High": 1,
            "Low": 0,
        },
    },
    "GPIO 3": {
        "value": 3,
        "choices": {
            "High": 1,
            "Low": 0,
        },
    },
    "GPIO 41": {
        "value": 41,
        "choices": {
            "High": 1,
            "Low": 0,
        },
    },
    "GPIO 42": {
        "value": 42,
        "choices": {
            "High": 1,
            "Low": 0,
        },
    },
}
STA_SAMPLE_INTERVAL_MS = 10
STA_ESTIMATED_TRAINING_MS = STA_SAMPLE_INTERVAL_MS * CONFIG_VALUES["window_size"]
STA_LAST_TRAINED = False
STA_IS_INVOKE = False
STA_POWER_ON_INVOKE = False
TAG_ID = 0
TAG_ID_LOCK = threading.Lock()
TASK_ID = 0
TASK_ID_LOCK = threading.Lock()
TASK_LOCK = threading.Lock()


# pylint: disable=missing-function-docstring
def at_request(
    command_name: str,
    command_args: list = None,
    wait: bool = False,
    timeout: float = 3.0,
    interval: float = 0.05,
):
    global TAG_ID, TAG_ID_LOCK, CLIENT  # pylint: disable=global-variable-not-assigned

    with TAG_ID_LOCK:
        TAG_ID += 1
        tag = TAG_ID

    tag = str(tag)
    command = f"AT+{tag}@{command_name}"
    if command_args is not None:
        if not isinstance(command_args, list):
            command_args = [command_args]
        command += "=" + ",".join(map(str, command_args))
    CLIENT.send_command(command)
    if not wait:
        return None

    start = time.time()
    while True:
        response = CLIENT.receive_response(ResponseType.Direct, command_name, tag)
        if response is not None:
            return response
        time.sleep(interval)
        if time.time() - start > timeout:
            break

    logging.error("Timeout while waiting for response to command: %s", command)
    return None


def wait_for_sensor_stream_response(current_task_id: int):
    global TASK_ID, CLIENT  # pylint: disable=global-variable-not-assigned

    while current_task_id == TASK_ID:
        response = CLIENT.receive_response(ResponseType.Stream, "AccelerometerData")
        if response is None:
            time.sleep(0.05)
            continue
        code = response.get("code", -1)
        if not isinstance(code, int) or code != 0:
            msg = f"Sensor stream response error: {response.get('msg', f'errorno {code}')}"
            logging.error(msg)
            gr.Warning(message=msg, duration=5)
            continue
        data = response.get("data", None)
        if not isinstance(data, dict):
            logging.error("No data received.")
            continue
        data_arr = data.get("data", None)
        if not isinstance(data_arr, list):
            logging.error("No data matrix received.")
            continue
        data_mat = []
        data_len = 0
        for row in data_arr:
            if not isinstance(row, list):
                logging.error("Invalid data row: %s", row)
                continue
            if data_len == 0:
                data_len = len(row)
            elif len(row) != data_len:
                logging.error(
                    "Row length mismatch: expected %d, got %d", data_len, len(row)
                )
                continue
            data_mat.append(np.array(row, dtype=np.float32))
        if len(data_mat) == 0 or data_len == 0:
            logging.error("No valid data rows received.")
            continue
        return {
            "len": data_len,
            "data": data_mat,
        }

    return None


def get_sensor_data(current_task_id: int):
    global TASK_ID, SENSOR_CACHE  # pylint: disable=global-variable-not-assigned

    response = wait_for_sensor_stream_response(current_task_id)
    if response is None:
        return None

    for i, row in enumerate(response["data"]):
        k = f"axis_{i}"
        SENSOR_CACHE["data"][k] = np.append(SENSOR_CACHE["data"][k], row)
    cache_limit = SENSOR_CACHE["limit"]
    for k, v in SENSOR_CACHE["data"].items():
        len_v = len(v)
        diff = cache_limit - len_v
        if diff > 0:
            SENSOR_CACHE["data"][k] = np.append(np.zeros(diff, dtype=v.dtype), v)
        elif diff < 0:
            SENSOR_CACHE["data"][k] = v[-cache_limit:]
    x_values = np.arange(cache_limit)

    return pd.concat(
        [
            pd.DataFrame({"x": x_values.copy(), "value": v.copy(), "curve": k})
            for k, v in SENSOR_CACHE["data"].items()
        ]
    )


def init_cleanup():
    global CLIENT  # pylint: disable=global-variable-not-assigned

    at_request("BREAK", wait=True)
    CLIENT.flush()
    CLIENT.clear_responses()


def parse_and_update_status(status: dict):
    global STA_LAST_TRAINED, STA_IS_INVOKE, STA_POWER_ON_INVOKE  # pylint: disable=global-statement

    last_trained = status.get("lastTrained", None)
    if isinstance(last_trained, bool):
        STA_LAST_TRAINED = last_trained
        logging.info("Last trained status updated: %s", STA_LAST_TRAINED)

    is_invoke = status.get("isInvoking", None)
    if isinstance(is_invoke, bool):
        STA_IS_INVOKE = is_invoke
        logging.info("Invoke status updated: %s", STA_IS_INVOKE)

    power_on_invoke = status.get("invokeOnStart", None)
    if isinstance(power_on_invoke, bool):
        STA_POWER_ON_INVOKE = power_on_invoke
        logging.info("Power on invoke status updated: %s", STA_POWER_ON_INVOKE)


def start_sensor_data_stream():
    res = at_request("START", ["sample"], wait=True)
    if res is None:
        logging.error("Start sensor data stream timed out or failed.")
        return False
    code = res.get("code", -1)
    if not isinstance(code, int) or code != 0:
        msg = f"Start sensor data stream response error: {res.get('msg', f'errorno {code}')}"
        logging.error(msg)
        gr.Warning(message=msg, duration=5)
        return False
    data = res.get("data")
    if not isinstance(data, dict):
        logging.error("Invalid data format received when starting sensor data stream.")
        return False
    status = data.get("status")
    if status is not None:
        parse_and_update_status(status)
    return True


def fetch_configs():  # pylint: disable=too-many-branches
    res = at_request("CFGGEDAD", wait=True)
    if res is None:
        logging.error("Fetch configs timed out or failed.")
        return False
    code = res.get("code", -1)
    if not isinstance(code, int) or code != 0:
        msg = f"Fetch configs response error: {res.get('msg', f'errorno {code}')}"
        logging.error(msg)
        return False
    data = res.get("data")
    if not isinstance(data, dict):
        logging.error("Invalid data format received for configs.")
        return False

    # pylint: disable=global-variable-not-assigned
    global \
        CONFIGS, \
        CONFIG_VALUES, \
        CONFIGURABLE_KEYS, \
        STA_SAMPLE_INTERVAL_MS, \
        STA_ESTIMATED_TRAINING_MS

    arg_array = data.get("args", None)
    if not isinstance(arg_array, list):
        logging.error("Invalid args format received for configs: %s", arg_array)
        return False
    len_configurable_keys = len(CONFIGURABLE_KEYS)
    if len(arg_array) >= len_configurable_keys:
        for k, v in zip(CONFIGURABLE_KEYS, arg_array[:len_configurable_keys]):
            if k not in CONFIGS:
                logging.error("Unknown config key: %s", k)
                continue
            if not isinstance(v, (int, float)):
                logging.error(
                    "Invalid value type for %s: %s, expected int or float",
                    k,
                    v,
                )
                continue
            cfg = CONFIGS[k]
            if cfg["min"] <= v <= cfg["max"]:
                if cfg["step"] == 1:
                    v = int(v)
                CONFIG_VALUES[k] = v
            else:
                logging.error(
                    "Invalid value for %s: %s not in [%s, %s]",
                    k,
                    v,
                    cfg["min"],
                    cfg["max"],
                )

    else:
        logging.error(
            "Invalid number of arguments received for configs. Expected >= %d, got %d",
            len_configurable_keys,
            len(arg_array),
        )

    sample_interval_ms = data.get("sampleInterval", None)
    if not isinstance(sample_interval_ms, int) or sample_interval_ms <= 0:
        logging.error(
            "Invalid sample interval: %s, last %s ms",
            sample_interval_ms,
            STA_SAMPLE_INTERVAL_MS,
        )
    else:
        STA_SAMPLE_INTERVAL_MS = sample_interval_ms

    estimated_sample_time = data.get("estimatedSampleTime", None)
    if not isinstance(estimated_sample_time, int) or estimated_sample_time <= 0:
        logging.error(
            "Invalid estimated sample time: %s, last %s ms",
            estimated_sample_time,
            STA_ESTIMATED_TRAINING_MS,
        )
    else:
        STA_ESTIMATED_TRAINING_MS = estimated_sample_time

    return True


def fetch_status():
    res = at_request("START", wait=True)
    if res is None:
        logging.error("Fetch status timed out or failed.")
        return False

    code = res.get("code", -1)
    if not isinstance(code, int) or code != 0:
        msg = f"Fetch status response error: {res.get('msg', f'errorno {code}')}"
        logging.error(msg)
        return False

    data = res.get("data", None)
    if not isinstance(data, dict):
        logging.error("Invalid data format received for status: %s", data)
        return False

    status = data.get("status", None)
    if not isinstance(status, dict):
        logging.error("Invalid status format received: %s", status)
        return False

    parse_and_update_status(status)

    return True


def update_sensor_data_stream():
    global TASK_ID, TASK_ID_LOCK, TASK_LOCK, CONFIG_VALUES  # pylint: disable=global-variable-not-assigned

    with TASK_ID_LOCK:
        TASK_ID += 1
        current_task_id = TASK_ID

    with TASK_LOCK:
        init_cleanup()
        while current_task_id == TASK_ID and not fetch_configs():
            time.sleep(0.1)
        while current_task_id == TASK_ID and not fetch_status():
            time.sleep(0.1)
        while current_task_id == TASK_ID and not start_sensor_data_stream():
            time.sleep(0.1)

        while current_task_id == TASK_ID:
            data = get_sensor_data(current_task_id)
            if data is not None:
                yield gr.update(value=data)
            else:
                time.sleep(0.1)

        yield None


def reset_client():
    yield gr.update(value="Resetting...", interactive=False)

    global STA_IS_INVOKE, CLIENT, TASK_ID, TASK_ID_LOCK  # pylint: disable=global-variable-not-assigned

    STA_IS_INVOKE = False

    CLIENT.send_command("")
    CLIENT.flush()

    at_request("RST", wait=False)
    time.sleep(1)

    init_cleanup()

    with TASK_ID_LOCK:
        current_task_id = TASK_ID

    while current_task_id == TASK_ID and not fetch_configs():
        time.sleep(0.1)
    while current_task_id == TASK_ID and not fetch_status():
        time.sleep(0.1)
    while current_task_id == TASK_ID and not start_sensor_data_stream():
        time.sleep(0.1)

    yield gr.update(value="Reset", interactive=True)


def train_model():  # pylint: disable=too-many-branches, too-many-statements
    yield gr.update(value="Requested...", interactive=False)

    # pylint: disable=global-variable-not-assigned
    global STA_IS_INVOKE, STA_ESTIMATED_TRAINING_MS, CLIENT, STA_LAST_TRAINED

    def reset():
        return gr.update(value="Train", interactive=True)

    STA_IS_INVOKE = False
    res = at_request("START", ["stop_invoke"], wait=True)
    if res is None:
        msg = "Timeout while stopping current inference task."
        logging.error(msg)
        gr.Warning(message=msg, duration=5)
        yield reset()
        return

    train_result = None
    res = at_request("TRAINGEDAD", ["train"], wait=True)
    if res is not None:
        tag = res.get("tag", "")
        code = res.get("code", -1)
        if isinstance(code, int) and code == 0:
            estimated = STA_ESTIMATED_TRAINING_MS / 1000.0
            spent = 0
            while spent < estimated:
                remain = max(0, estimated - spent)
                yield gr.update(
                    value=f"Estimated: {remain:.1f}s",
                    interactive=False,
                )
                time.sleep(0.1)
                spent += 0.1
            start = time.time()
            while True:
                train_result = CLIENT.receive_response(
                    ResponseType.Event, "TRAINGEDAD", tag
                )
                if train_result is not None:
                    break
                time.sleep(0.1)
                if time.time() - start > 5:
                    break

        else:
            msg = f"Training response error: {res.get('msg', f'errorno {code}')}"
            logging.error(msg)
            gr.Warning(message=msg, duration=5)

    if train_result is None:
        msg = "Timeout while waiting for training response."
        logging.error(msg)
        gr.Warning(message=msg, duration=5)
        yield reset()
        return

    code = train_result.get("code", -1)
    if isinstance(code, int) and code == 0:
        STA_LAST_TRAINED = True
        data = train_result.get("data", None)
        if isinstance(data, dict):
            euclidean_dist_thresh = data.get("euclideanDistThresh", None)
            contiguous_n = data.get("contiguousN", None)
            # pylint: disable=line-too-long
            msg = f"Training completed, euclidean distance threshold: {euclidean_dist_thresh}, contiguous N: {contiguous_n}"
            logging.info(msg)
            gr.Info(message=msg, duration=5)
            yield gr.update(value="Success", interactive=False)
            time.sleep(0.5)
    else:
        msg = f"Training response error: {train_result.get('msg', f'errorno {code}')}"
        logging.error(msg)
        gr.Warning(message=msg, duration=5)

    yield reset()


def save_model():
    yield gr.update(value="Saving...", interactive=False)

    global STA_LAST_TRAINED  # pylint: disable=global-variable-not-assigned

    def reset():
        return gr.update(value="Save", interactive=True)

    global STA_LAST_TRAINED  # pylint: disable=global-variable-not-assigned

    if not STA_LAST_TRAINED:
        logging.error("Model has not been trained yet, please train the model first.")
        yield gr.update(value="Please Train First", interactive=False)
        time.sleep(0.5)
        yield reset()
        return

    res = at_request("TRAINGEDAD", ["save"], wait=True)
    if res is not None:
        code = res.get("code", -1)
        if isinstance(code, int) and code == 0:
            logging.info("Model saved successfully.")
            yield gr.update(value="Success", interactive=False)
            time.sleep(0.5)
        else:
            msg = f"Failed to save model: {res.get('msg', f'errorno {code}')}"
            logging.error(msg)
            gr.Warning(message=msg, duration=5)
    else:
        msg = "Timeout while waiting for save model response."
        logging.error(msg)
        gr.Warning(message=msg, duration=5)

    yield reset()


def get_inference_data(current_task_id: int):  # pylint: disable=too-many-branches
    global TASK_ID, STA_IS_INVOKE, CLIENT  # pylint: disable=global-variable-not-assigned

    res = {
        "Anomaly": 0.0,
        "Normal": 0.0,
    }

    while current_task_id == TASK_ID and STA_IS_INVOKE:
        response = CLIENT.receive_response(ResponseType.Stream, "ClassifyResult")
        if response is None:
            time.sleep(0.05)
            continue
        code = response.get("code", -1)
        if not isinstance(code, int) or code != 0:
            STA_IS_INVOKE = False
            res["Anomaly"] = 1.0
            msg = f"Inference response error: {response.get('msg', f'errorno {code}')}"
            logging.error(msg)
            gr.Warning(message=msg, duration=5)
            break
        data = response.get("data", None)
        if not isinstance(data, dict):
            logging.error("No classify data received.")
            continue
        data_arr = data.get("data", None)
        if not isinstance(data_arr, list):
            logging.error("No classify data matrix received.")
            continue
        if len(data_arr) != 2:
            logging.error(
                "Invalid classify data length: expected 2, got %d",
                len(data_arr),
            )
            continue

        for cls in data_arr:
            if not isinstance(cls, list):
                logging.error("Invalid classify data row: %s", cls)
                continue
            if len(cls) != 2:
                logging.error(
                    "Invalid classify data row length: expected 2, got %d",
                    len(cls),
                )
                continue
            cls_id = cls[0]
            if not isinstance(cls_id, int):
                logging.error("Invalid classify data class: %s", cls_id)
                continue
            cls_conf = cls[1]
            if not isinstance(cls_conf, (int, float)):
                logging.error("Invalid classify data confidence: %s", cls_conf)
                continue

            if cls_id == 0:
                res["Normal"] = cls_conf
            elif cls_id == 1:
                res["Anomaly"] = cls_conf
            else:
                logging.error("Invalid classify data class: %s", cls_id)
                continue

        break

    return res


def update_inference_stream():
    yield None, gr.update(value="Requested...", interactive=False)

    global STA_LAST_TRAINED, STA_IS_INVOKE, TASK_ID, TASK_ID_LOCK  # pylint: disable=global-variable-not-assigned

    def reset():
        return gr.update(value="Inference", interactive=True)

    if not STA_LAST_TRAINED:
        logging.error("Model has not been trained yet, please train the model first.")
        STA_IS_INVOKE = False
        yield (
            None,
            gr.update(value="Please Train First", interactive=False),
        )
        time.sleep(0.5)
        yield None, reset()
        return

    res = at_request("START", ["invoke"], wait=True)
    if res is not None:
        code = res.get("code", -1)
        if isinstance(code, int) and code == 0:
            STA_IS_INVOKE = True
            logging.info("Inference started successfully.")
        else:
            STA_IS_INVOKE = False
            msg = f"Start inference response error: {res.get('msg', f'errorno {code}')}"
            logging.error(msg)
            gr.Warning(message=msg, duration=5)
            yield None, reset()
            return
    else:
        STA_IS_INVOKE = False
        msg = "Timeout while starting inference task."
        logging.error(msg)
        gr.Warning(message=msg, duration=5)
        yield None, reset()
        return

    yield None, gr.update(value="Running...", interactive=False)

    with TASK_ID_LOCK:
        current_task_id = TASK_ID

    while current_task_id == TASK_ID and STA_IS_INVOKE:
        yield get_inference_data(current_task_id), gr.update()

    yield None, reset()


def validate_config_constraints(config_name: str, value: int | float):
    # pylint: disable=global-variable-not-assigned
    global CONFIG_VALUES, CONFIG_RELATIVE_CONSTRAINTS

    for constraint in CONFIG_RELATIVE_CONSTRAINTS:
        related = constraint["related"]
        if config_name not in related:
            continue
        x = {
            k: CONFIG_VALUES.get(k, None) if k != config_name else value
            for k in related
        }
        if not constraint["expr"](x):
            return constraint["msg"]

    return None


# pylint: disable=too-many-return-statements
def update_config_values(
    config_name: str, value: int | float, dynamic_exception: list = None
):
    # pylint: disable=global-variable-not-assigned
    global \
        CONFIGURABLE_KEYS, \
        CONFIGURABLE_KEYS_DYNAMIC, \
        STA_IS_INVOKE, \
        CONFIGS, \
        CONFIG_VALUES

    val_rng = CONFIGS.get(config_name, None)
    val_lst = CONFIG_VALUES.get(config_name, val_rng.get("default", None))
    if not isinstance(val_rng, dict) or val_lst is None:
        logging.error("Config key %s is not configurable.", config_name)
        return val_lst
    if not val_rng["min"] <= value <= val_rng["max"]:
        logging.error(
            "Config value %s is out of range [%s, %s].",
            value,
            val_rng["min"],
            val_rng["max"],
        )
        return val_lst

    try:
        violated = validate_config_constraints(config_name, value)
        if violated is not None:
            msg = f"Config value {value} violates constraints: {violated}"
            logging.error(msg)
            gr.Warning(message=msg, duration=5)
            return val_lst
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error validating config constraints: %s", e)
        return val_lst

    is_dynamic = config_name in CONFIGURABLE_KEYS_DYNAMIC or (
        dynamic_exception is not None and config_name in dynamic_exception
    )
    if not is_dynamic:
        STA_IS_INVOKE = False
        res = at_request("START", ["stop_invoke"], wait=True)
        if res is None:
            msg = "Timeout while stopping current inference task."
            logging.error(msg)
            gr.Warning(message=msg, duration=5)
            return val_lst

    cmd_args = []
    for k in CONFIGURABLE_KEYS:
        if k == config_name:
            cmd_args.append(value)
            break
        cmd_args.append("")

    res = at_request("CFGGEDAD", cmd_args, wait=True)
    if res is not None:
        code = res.get("code", -1)
        if isinstance(code, int) and code == 0:
            CONFIG_VALUES[config_name] = value
            logging.info("Config %s updated to %s.", config_name, value)
            if not is_dynamic and not fetch_configs():
                logging.error("Failed to fetch update configs.")
            return CONFIG_VALUES[config_name]

        msg = f"Update config response error: {res.get('msg', f'errorno {code}')}"
        logging.error(msg)
        gr.Warning(message=msg, duration=5)
        return val_lst

    msg = "Timeout while updating config values."
    logging.error(msg)
    gr.Warning(message=msg, duration=5)
    return val_lst


def build_config_slider_callbacks(config_name: str):
    return lambda value: update_config_values(config_name, value)


def get_gpio_output_key():
    global CONFIGURABLE_GPIOS, CONFIG_VALUES  # pylint: disable=global-variable-not-assigned

    if len(CONFIGURABLE_GPIOS) == 0:
        return None

    key = list(CONFIGURABLE_GPIOS.keys())[0]
    val = CONFIG_VALUES["abnormal_output_gpio"]
    for k, v in CONFIGURABLE_GPIOS.items():
        if v["value"] == val:
            key = k
            break
    return key


def get_gpio_output_choices(key: str):
    global CONFIGURABLE_GPIOS  # pylint: disable=global-variable-not-assigned

    return CONFIGURABLE_GPIOS[key]["choices"]


def get_gpio_output_choice_selected(choices: dict):
    global CONFIG_VALUES  # pylint: disable=global-variable-not-assigned

    key = list(choices.keys())[0] if len(choices) > 0 else None
    val = CONFIG_VALUES["abnormal_output_gpio_value"]
    for k, v in choices.items():
        if v == val:
            key = k
            break
    return key


def update_gpio_output_selector(key: str):
    choices = get_gpio_output_choices(key)
    selected = get_gpio_output_choice_selected(choices)

    excepted_value = CONFIGURABLE_GPIOS[key]["value"]
    actual_value = update_config_values(
        "abnormal_output_gpio",
        excepted_value,
        dynamic_exception=["abnormal_output_gpio"],
    )
    if actual_value != excepted_value:
        return gr.update(), gr.update()

    return gr.update(value=key), gr.update(
        choices=list(choices.keys()), value=selected, interactive=len(choices) > 1
    )


def update_gpio_output_value_selector(key: str, value: str):
    global CONFIGURABLE_GPIOS, CONFIG_VALUES  # pylint: disable=global-variable-not-assigned

    choices = get_gpio_output_choices(key)
    if len(choices) == 0:
        return gr.update(interactive=False)
    selected = get_gpio_output_choice_selected(choices)
    if value == selected:
        logging.info("GPIO output value is already set to %s.", value)
        return gr.update(value=value)

    excepted_value = choices.get(value, None)
    if excepted_value is None:
        logging.error("Invalid GPIO output value: %s", value)
        return gr.update()
    actual_value = update_config_values(
        "abnormal_output_gpio_value",
        excepted_value,
        dynamic_exception=["abnormal_output_gpio_value"],
    )
    if actual_value != excepted_value:
        return gr.update()

    return gr.update(value=value)


def update_power_on_invoke(value: str):
    global STA_POWER_ON_INVOKE  # pylint: disable=global-statement

    power_on_invoke = value == "Enabled"
    if power_on_invoke == STA_POWER_ON_INVOKE:
        logging.info("Power On Inference is already set to %s.", value)
        return gr.update(value=value)

    res = at_request("START", ["enable" if power_on_invoke else "disable"], wait=True)
    if res is None:
        msg = "Timeout while updating power on inference status."
        logging.error(msg)
        gr.Warning(message=msg, duration=5)
        return gr.update(value=STA_POWER_ON_INVOKE)

    code = res.get("code", -1)
    if not isinstance(code, int) or code != 0:
        msg = f"Update power on inference response error: {res.get('msg', f'errorno {code}')}"
        logging.error(msg)
        gr.Warning(message=msg, duration=5)
        return gr.update(value=STA_POWER_ON_INVOKE)

    STA_POWER_ON_INVOKE = power_on_invoke
    logging.info("Power On Inference updated to %s.", value)

    return gr.update(value=value)


def build_demo():  # pylint: disable=too-many-locals,too-many-statements
    # pylint: disable=global-variable-not-assigned
    global \
        CLIENT, \
        SENSOR_CACHE, \
        CONFIGS, \
        CONFIG_VALUES, \
        CONFIGURABLE_KEYS_STATIC, \
        CONFIGURABLE_KEYS_DYNAMIC, \
        CONFIGURABLE_GPIOS, \
        STA_POWER_ON_INVOKE

    with gr.Blocks(title="AcousticsLab", delete_cache=[60, 60]) as demo:
        demo.queue(max_size=2, default_concurrency_limit=2)

        sensor_output = gr.LinePlot(
            x="x",
            y="value",
            color="curve",
            x_lim=[0, SENSOR_CACHE["limit"]],
            height=500,
            label="Accelerometer Data",
            # buttons=["fullscreen"],  # Disabled due to Gradio 6.5.1 bug
            show_label=True,
        )

        # pylint: disable=no-member
        demo.load(
            fn=update_sensor_data_stream,
            inputs=None,
            outputs=sensor_output,
        )

        config_sliders = {}

        with gr.Row():
            with gr.Column():
                for key in CONFIGURABLE_KEYS_STATIC:
                    config = CONFIGS[key]
                    config_value = CONFIG_VALUES[key]
                    config_sliders[key] = gr.Slider(
                        minimum=config["min"],
                        maximum=config["max"],
                        value=config_value,
                        step=config["step"],
                        label=f"{' '.join([v if v in ('at', 'in', 'of') else v.capitalize() for v in key.split('_')])}",  # pylint: disable=line-too-long
                        interactive=True,
                    )

                with gr.Row():
                    gpio_output_key = get_gpio_output_key()
                    gpio_output_selector = gr.Dropdown(
                        choices=CONFIGURABLE_GPIOS.keys(),
                        value=gpio_output_key,
                        label="Abnormal Output",
                        interactive=True,
                    )

                    gpio_output_value_choices = get_gpio_output_choices(gpio_output_key)
                    gpio_output_value_selector = gr.Dropdown(
                        choices=list(gpio_output_value_choices.keys()),
                        value=get_gpio_output_choice_selected(
                            gpio_output_value_choices
                        ),
                        interactive=len(gpio_output_value_choices) > 1,
                        label="Output Mode",
                    )
                    gpio_output_selector.input(
                        update_gpio_output_selector,
                        inputs=gpio_output_selector,
                        outputs=[gpio_output_selector, gpio_output_value_selector],
                    )
                    gpio_output_value_selector.input(
                        update_gpio_output_value_selector,
                        inputs=[gpio_output_selector, gpio_output_value_selector],
                        outputs=gpio_output_value_selector,
                    )

                    power_on_invoke_selector = gr.Dropdown(
                        choices=["Enabled", "Disabled"],
                        value="Enabled" if STA_POWER_ON_INVOKE else "Disabled",
                        label="Power On Inference",
                        interactive=True,
                    )
                    power_on_invoke_selector.input(
                        update_power_on_invoke,
                        inputs=power_on_invoke_selector,
                        outputs=power_on_invoke_selector,
                    )

            with gr.Column():
                with gr.Row():
                    reset_button = gr.Button("Reset", variant="stop")
                    reset_button.click(
                        fn=reset_client, inputs=None, outputs=reset_button
                    )
                    train_button = gr.Button("Train", variant="primary")
                    train_button.click(
                        fn=train_model, inputs=None, outputs=train_button
                    )
                    save_button = gr.Button("Save", variant="secondary")
                    save_button.click(fn=save_model, inputs=None, outputs=save_button)

                inference_output = gr.Label(
                    num_top_classes=2,
                    label="Result",
                )

                for key in CONFIGURABLE_KEYS_DYNAMIC:
                    config = CONFIGS[key]
                    config_value = CONFIG_VALUES[key]
                    config_sliders[key] = gr.Slider(
                        minimum=config["min"],
                        maximum=config["max"],
                        value=config_value,
                        step=config["step"],
                        label=f"{' '.join([v if v in ('at', 'in', 'of') else v.capitalize() for v in key.split('_')])}",  # pylint: disable=line-too-long
                        interactive=True,
                    )

                inference_button = gr.Button("Inference", variant="primary")
                inference_button.click(
                    fn=update_inference_stream,
                    inputs=None,
                    outputs=[inference_output, inference_button],
                )

        for key, slider in config_sliders.items():
            slider.release(
                fn=build_config_slider_callbacks(key), inputs=slider, outputs=slider
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Gradio service for real-time plotting and device control."
    )
    parser.add_argument(
        "--device",
        default="/dev/ttyACM0",
        help="Serial device device name (e.g. /dev/ttyACM0)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=921600,
        help="Baud rate for the serial communication (e.g. 921600)",
    )
    parser.add_argument(
        "--data-buffer-limit",
        type=int,
        default=8192,
        help="Maximum data buffer limit",
    )
    parser.add_argument(
        "--response-buffer-limit",
        type=int,
        default=32,
        help="Maximum response buffer limit",
    )
    parser.add_argument(
        "--share", default=False, action="store_true", help="Allow public access"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    DEMO = None
    CLIENT_THREAD = None
    try:
        logging.info(
            "Starting client on device %s with baudrate %s...",
            args.device,
            args.baudrate,
        )
        CLIENT = Client(
            device=args.device,
            baudrate=args.baudrate,
            data_buffer_limit=args.data_buffer_limit,
            response_buffer_limit=args.response_buffer_limit,
        )
        CLIENT_THREAD = threading.Thread(target=CLIENT.run, daemon=True)
        CLIENT_THREAD.start()

        init_cleanup()
        while not fetch_configs():
            time.sleep(0.1)
        while not fetch_status():
            time.sleep(0.1)

        logging.info("Starting gradio service...")
        DEMO = build_demo()
        DEMO.launch(share=args.share)
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("An error occurred: %s", e)
    finally:
        if CLIENT is not None:
            CLIENT.stop()
        if CLIENT_THREAD is not None:
            CLIENT_THREAD.join()
        if DEMO is not None:
            DEMO.close()
