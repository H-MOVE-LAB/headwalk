from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def strip_quantization_config(obj):
    """
    Recursively remove quantization_config fields from a Keras config object.

    This is useful when a model was saved with a newer Keras version that writes
    quantization_config=None into layer configs, while an older Keras version
    cannot deserialize that keyword.
    """

    if isinstance(obj, dict):
        return {
            key: strip_quantization_config(value)
            for key, value in obj.items()
            if key != "quantization_config"
        }

    if isinstance(obj, list):
        return [strip_quantization_config(item) for item in obj]

    return obj


def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def try_step(name, func):
    print_header(name)

    try:
        result = func()
        print("[OK]", name)
        return True, result

    except Exception as exc:
        print("[FAIL]", name)
        print(type(exc).__name__ + ":", exc)
        print("\n--- traceback ---")
        traceback.print_exc(limit=8)
        return False, None


def main() -> None:
    print_header("ENVIRONMENT")

    print("python:", sys.version)
    print("python executable:", sys.executable)

    try:
        import tensorflow as tf
        print("tensorflow:", tf.__version__)
        print("tensorflow file:", tf.__file__)
    except Exception as exc:
        print("tensorflow import failed:", repr(exc))

    try:
        import keras
        print("keras:", keras.__version__)
        print("keras file:", keras.__file__)
    except Exception as exc:
        print("keras import failed:", repr(exc))

    model_root = PROJECT_ROOT / "src" / "headwalk" / "gait_sequence_detection" / "models"

    keras_path = find_first_existing([
        model_root / "GsdCnn1D.keras",
        model_root / "portable_artifacts" / "GsdCnn1D" / "model.keras",
        model_root / "portable_artifacts" / "GsdCnn1D" / "GsdCnn1D.keras",
    ])

    weights_path = find_first_existing([
        model_root / "GsdCnn1D.weights.h5",
        model_root / "portable_artifacts" / "GsdCnn1D" / "model.weights.h5",
        model_root / "portable_artifacts" / "GsdCnn1D" / "GsdCnn1D.weights.h5",
    ])

    json_path = find_first_existing([
        model_root / "GsdCnn1D.json",
        model_root / "portable_artifacts" / "GsdCnn1D" / "model.json",
        model_root / "portable_artifacts" / "GsdCnn1D" / "GsdCnn1D.json",
    ])

    print_header("CNN ARTIFACTS")

    for label, path in [
        ("keras_path", keras_path),
        ("weights_path", weights_path),
        ("json_path", json_path),
    ]:
        print(label + ":", path)
        if path is not None and path.exists():
            print("  size:", path.stat().st_size, "bytes")

    def init_headwalk_class():
        from headwalk.gait_sequence_detection.algo import GsdCnn1D

        model = GsdCnn1D()
        print("class:", type(model))
        return model

    try_step("HEADWALK GsdCnn1D() INITIALIZATION", init_headwalk_class)

    def direct_keras_load():
        if keras_path is None:
            raise FileNotFoundError("No .keras CNN artifact found.")

        import keras

        model = keras.models.load_model(
            keras_path,
            compile=False,
            safe_mode=False,
        )
        print(model.summary())
        return model

    try_step("DIRECT keras.models.load_model(.keras)", direct_keras_load)

    def direct_tf_keras_load():
        if keras_path is None:
            raise FileNotFoundError("No .keras CNN artifact found.")

        import tensorflow as tf

        model = tf.keras.models.load_model(
            keras_path,
            compile=False,
            safe_mode=False,
        )
        print(model.summary())
        return model

    try_step("DIRECT tf.keras.models.load_model(.keras)", direct_tf_keras_load)

    def patched_keras_zip_load():
        if keras_path is None:
            raise FileNotFoundError("No .keras CNN artifact found.")

        import keras

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            patched_path = tmp_dir / "GsdCnn1D_patched.keras"

            with zipfile.ZipFile(keras_path, "r") as zin:
                with zipfile.ZipFile(patched_path, "w") as zout:
                    for item in zin.infolist():
                        data = zin.read(item.filename)

                        if item.filename == "config.json":
                            config = json.loads(data.decode("utf-8"))
                            config = strip_quantization_config(config)
                            data = json.dumps(config).encode("utf-8")

                        zout.writestr(item, data)

            model = keras.models.load_model(
                patched_path,
                compile=False,
                safe_mode=False,
            )
            print(model.summary())
            return model

    try_step("PATCHED .keras LOAD WITHOUT quantization_config", patched_keras_zip_load)

    def json_plus_weights_load():
        if json_path is None:
            raise FileNotFoundError("No CNN json architecture file found.")

        if weights_path is None:
            raise FileNotFoundError("No CNN weights file found.")

        import keras

        raw_config = json.loads(json_path.read_text(encoding="utf-8"))
        clean_config = strip_quantization_config(raw_config)
        model_json = json.dumps(clean_config)

        model = keras.models.model_from_json(model_json)
        model.load_weights(weights_path)

        print(model.summary())
        return model

    try_step("JSON ARCHITECTURE + .weights.h5 LOAD", json_plus_weights_load)


if __name__ == "__main__":
    main()
