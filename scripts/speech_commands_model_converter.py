import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional
import logging
import mimetypes
import random
import tarfile
import urllib.request
import json

import librosa
import numpy as np
import tensorflow as tf
from tensorflowjs.converters.converter import (
    dispatch_tensorflowjs_to_keras_saved_model_conversion,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

CWD = Path.cwd()
DEFAULT_TFJS_MODEL = CWD / "sensecraft-audio-model-2025-11-13"
DEFAULT_WAV_INPUT_DIR = CWD / "calibration_audio"
DEFAULT_OUT_MODEL = CWD / "sensecraft-audio-model-2025-11-13.tflite"
REQUIRED_SAMPLES = 100


def _download_and_extract_preproc_model(dest_dir: Path) -> Path:
    PREPROC_MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/conversion/sc_preproc_model.tar.gz"
    MODEL_DIR_NAME = "sc_preproc_model"
    model_path = dest_dir / MODEL_DIR_NAME

    if model_path.exists():
        logging.info(f"Preprocessing model found at {model_path}")
        return model_path

    logging.info(f"Downloading preprocessing model from {PREPROC_MODEL_URL}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        tar_path, _ = urllib.request.urlretrieve(PREPROC_MODEL_URL)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        os.remove(tar_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download or extract preprocessing model: {e}"
        ) from e

    logging.info(f"Successfully extracted preprocessing model to {model_path}")
    return model_path


def _resolve_tfjs_input_path(tfjs_path: Path) -> Path:
    if tfjs_path.is_dir():
        cand = tfjs_path / "model.json"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"model.json not found in {tfjs_path}")
    if tfjs_path.name == "model.json" and tfjs_path.exists():
        return tfjs_path
    raise FileNotFoundError(
        f"Expected a TFJS layers model directory (with model.json) or a model.json file, got: {tfjs_path}"
    )

def _run_tfjs_to_saved_model(tfjs_model_path_or_dir: Path, saved_model_out_dir: Path):
    original_model_json_path = _resolve_tfjs_input_path(tfjs_model_path_or_dir)
    
    with open(original_model_json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    model_json_to_convert = None
    temp_file_name = None

    try:
        if "modelTopology" in model_data and "weightsManifest" in model_data:
            logging.info("Detected combined TFJS model format.")
            model_json_to_convert = original_model_json_path
        else:
            logging.info("Detected separated TFJS model format. Combining files.")
            
            model_topology = model_data
            
            weights_manifest_path = original_model_json_path.parent / "weights_manifest.json"
            if not weights_manifest_path.exists():
                weights_manifest_path = original_model_json_path.parent / "manifest.json"
                if not weights_manifest_path.exists():
                     raise FileNotFoundError(
                        f"Could not find 'weights_manifest.json' or 'manifest.json' in {original_model_json_path.parent}"
                    )

            with open(weights_manifest_path, "r", encoding="utf-8") as f:
                weight_specs = json.load(f)
            
            original_weights_bin_path = (original_model_json_path.parent / "weights.bin").resolve()
            if not original_weights_bin_path.is_file():
                raise FileNotFoundError(f"The weights file 'weights.bin' was not found in {original_model_json_path.parent}")
            
            formatted_weights_manifest = [
                {
                    "paths": [str(original_weights_bin_path)],
                    "weights": weight_specs
                }
            ]
            
            combined_model_data = {
                "modelTopology": model_topology,
                "weightsManifest": formatted_weights_manifest
            }
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json", encoding="utf-8") as temp_file:
                json.dump(combined_model_data, temp_file)
                temp_file_name = temp_file.name
            
            model_json_to_convert = temp_file_name
            logging.info(f"Created temporary combined model file: {model_json_to_convert}")
            logging.info(f"Pointing to weights file at: {original_weights_bin_path}")

        saved_model_out_dir.parent.mkdir(parents=True, exist_ok=True)
        dispatch_tensorflowjs_to_keras_saved_model_conversion(
            str(model_json_to_convert),
            str(saved_model_out_dir),
        )
    finally:
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)
            logging.info(f"Cleaned up temporary model file: {temp_file_name}")

    if not saved_model_out_dir.exists():
        print(f"ERROR: SavedModel not found at {saved_model_out_dir}", file=sys.stderr)
        raise FileNotFoundError(saved_model_out_dir)

def _make_representative_dataset_generator(
    input_shape: tuple[int, ...], wav_dir: Optional[Path]
) -> Callable[[], Iterable[list]]:
    def generator() -> Iterable[list]:
        if not wav_dir or not wav_dir.is_dir():
            raise FileNotFoundError(
                f"For 'int8' quantization, a directory with WAV files is required. "
                f"Please provide a valid path using the --wav-dir argument. Path given: {wav_dir}"
            )
        logging.info("Generating features from audio for INT8 quantization")
        with tempfile.TemporaryDirectory() as tmpdir:
            preproc_model_path = _download_and_extract_preproc_model(Path(tmpdir))
            preproc_model = tf.saved_model.load(str(preproc_model_path))
            inference_func = preproc_model.signatures["serving_default"]

        audio_files = [
            p
            for p in wav_dir.iterdir()
            if p.is_file()
            and (mtype := mimetypes.guess_type(p)[0])
            and mtype.startswith("audio/")
        ]

        if not audio_files:
            raise FileNotFoundError(f"No audio files found in directory: {wav_dir}")

        logging.info(f"Found {len(audio_files)} audio files.")
        num_samples = min(len(audio_files), REQUIRED_SAMPLES)
        selected_files = random.sample(audio_files, num_samples)
        logging.info(f"Using {len(selected_files)} files for calibration.")

        target_len = 44032
        target_sr = 44100

        for audio_path in selected_files:
            try:
                data, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
                if len(data) > target_len:
                    start = (len(data) - target_len) // 2
                    data = data[start : start + target_len]
                else:
                    data = np.pad(data, (0, target_len - len(data)), "constant")

                result = inference_func(
                    tf.constant(data.reshape(1, target_len), dtype=tf.float32)
                )
                feature_map = list(result.values())[0]

                if feature_map.shape != input_shape:
                    logging.warning(
                        f"Feature shape {feature_map.shape} != model shape {input_shape}. Skipping {audio_path.name}"
                    )
                    continue

                yield [feature_map.numpy().astype(np.float32)]
            except Exception as e:
                logging.warning(
                    f"Could not process '{audio_path.name}': {e}. Skipping."
                )
                continue

    return generator


def _get_concrete_input_shape(model: tf.Module) -> Optional[tuple[int, ...]]:
    concrete = None
    signatures = model.signatures
    if "serving_default" in signatures:
        concrete = signatures["serving_default"]
    elif len(signatures) > 0:
        concrete = next(iter(signatures.values()))
    if concrete is not None and concrete.inputs:
        tshape = concrete.inputs[0].shape.as_list()
        if tshape and tshape[0] is None:
            tshape[0] = 1
        return tuple(int(x) for x in tshape)
    return None


def convert(
    tfjs_dir: Path,
    output_tflite: Path,
    quantize: str,
    wav_dir: Optional[Path] = None,
    input_shape_override: Optional[str] = None,
) -> Path:
    tfjs_dir = tfjs_dir.resolve()
    if not (tfjs_dir / "model.json").exists():
        raise FileNotFoundError(f"model.json not found in {tfjs_dir}")
    output_tflite = output_tflite.resolve()
    converter = None
    model = None
    quantize = quantize.lower().strip()
    with tempfile.TemporaryDirectory(prefix="tfjs2tflite_") as tmpdir:
        saved_model_dir = Path(tmpdir) / "saved_model"

    try:
        _run_tfjs_to_saved_model(tfjs_dir, saved_model_dir)
        model = tf.saved_model.load(str(saved_model_dir))

        match quantize:
            case "dynamic":
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    str(saved_model_dir)
                )

            case "float16":
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    str(saved_model_dir)
                )
                converter.target_spec.supported_types = [tf.float16]

            case "int8":
                input_shape = _get_concrete_input_shape(model)
                if input_shape_override:
                    dims = tuple(
                        int(x.strip())
                        for x in input_shape_override.split(",")
                        if x.strip()
                    )
                    if len(dims) == 0:
                        raise ValueError("--input-shape provided but empty")
                    if dims[0] != 1:
                        dims = (1,) + dims
                    input_shape = dims
                if input_shape is None:
                    raise ValueError("Could not automatically infer model input shape")
                concrete_func = model.signatures[
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                ]
                concrete_func.inputs[0].set_shape(input_shape)
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_func]
                )
                converter.representative_dataset = (
                    _make_representative_dataset_generator(input_shape, wav_dir)
                )
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

            case _:
                raise ValueError(f"Unsupported quantization mode: {quantize}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to convert TFJS model to TFLite model due to: {e}"
        ) from e

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tflite, "wb") as f:
        f.write(tflite_model)

    return output_tflite


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Speech Commands TFJS model to TFLite"
    )
    p.add_argument(
        "--tfjs-dir",
        type=Path,
        default=DEFAULT_TFJS_MODEL,
        help="Directory with model.json and weights.bin",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT_MODEL,
        help="Output .tflite file path",
    )
    p.add_argument(
        "--quantize",
        choices=["dynamic", "float16", "int8"],
        default="int8",
        help="Quantization mode (default: int8)",
    )
    p.add_argument(
        "--wav-dir",
        type=Path,
        default=DEFAULT_WAV_INPUT_DIR,
        help="Directory of .wav files for 'int8' quantization calibration. If not provided, random data will be used.",
    )
    p.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Comma-separated input shape excluding batch, e.g. '43,232,1'",
    )

    args = p.parse_args(argv)

    try:
        logging.info("Starting Model Conversion Pipeline")
        logging.info(f"   - TF.js Source: {args.tfjs_dir}")
        logging.info(f"   - TFLite Output: {args.output}")
        logging.info(f"   - Quantization: {args.quantize}")
        if args.quantize == "int8":
            logging.info(f"   - Calibration Data: {args.wav_dir}")

        out = convert(
            args.tfjs_dir,
            args.output,
            args.quantize,
            wav_dir=args.wav_dir,
            input_shape_override=args.input_shape,
        )
        size_kb = os.path.getsize(out) / 1024.0
        logging.info("=" * 50)
        logging.info("Conversion successful!")
        logging.info(f"   - Output: {out} ({size_kb:.1f} KB)")
        logging.info("=" * 50)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"ERROR: {e}")
        return 1
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
