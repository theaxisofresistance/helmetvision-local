"""
Flask application for helmet detection.

This application provides a simple web interface for uploading an image and
running it through a pre‑trained object detection model to determine whether
helmets are present.  The core logic is adapted from open source examples
where YOLOv5 is loaded via ``torch.hub`` and the resulting annotated image
is returned to the user.  A results directory under ``static/results`` is
created automatically to store annotated images.

Note: To actually perform inference, you must supply a trained model
weights file (e.g., ``best.pt``) and ensure PyTorch is installed.  The
default path for the weights file is ``weights/best.pt`` relative to the
application root, but this can be overridden by setting the ``MODEL_WEIGHTS``
environment variable.  If the weights are not found or PyTorch isn't
installed, the app will still run but will not annotate images.
"""

import os
import tempfile
from functools import lru_cache
from flask import Flask, jsonify, request, render_template, url_for
from werkzeug.utils import secure_filename

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

try:
    import torch  # type: ignore
except ImportError:
    # PyTorch is optional; the app can still run without performing inference
    torch = None  # type: ignore


IMG_SIZE_ENV = os.environ.get("IMG_SIZE", "128,128")
IMG_SIZE = tuple(int(part.strip()) for part in IMG_SIZE_ENV.split(",", 1))
FEATURE_SIZE_ENV = os.environ.get("FEATURE_IMAGE_SIZE", "262,496")
FEATURE_IMAGE_SIZE = tuple(int(part.strip()) for part in FEATURE_SIZE_ENV.split(",", 1))
FEATURE_COLOR_MODE = os.environ.get("FEATURE_COLOR_MODE", "grayscale").lower()
USE_PCA = True
MODEL_BUNDLE_PATH = os.environ.get("MODEL_BUNDLE_PATH", "models.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")
PCA_PATH = os.environ.get("PCA_PATH", "pca.pkl")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", "encoder.pkl")


def load_joblib_file(path: str):
    """Load a joblib artifact from disk."""
    if joblib is None:
        raise ImportError("joblib is not installed in this environment.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_models(model_path: str = MODEL_BUNDLE_PATH):
    """Load the bundled sklearn models."""
    return load_joblib_file(model_path)


@lru_cache(maxsize=1)
def load_scaler(path: str = SCALER_PATH):
    """Load the fitted feature scaler."""
    return load_joblib_file(path)


@lru_cache(maxsize=1)
def load_pca(path: str = PCA_PATH):
    """Load the fitted PCA transformer."""
    return load_joblib_file(path)


@lru_cache(maxsize=1)
def load_label_encoder(path: str = LABEL_ENCODER_PATH):
    """Load the label encoder used during training."""
    return load_joblib_file(path)


def describe_joblib_loading_error(exc: Exception) -> str:
    """Convert artifact loading failures into a readable readiness message."""
    if isinstance(exc, ModuleNotFoundError):
        return f"Missing Python dependency required by joblib artifacts: {exc.name}."
    return f"Could not load joblib ML artifacts: {exc}"


def extract_features(img):
    """Convert an image into the same flattened feature shape used in training."""
    resized = cv2.resize(img, (FEATURE_IMAGE_SIZE[1], FEATURE_IMAGE_SIZE[0]))

    if FEATURE_COLOR_MODE == "grayscale":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    elif FEATURE_COLOR_MODE != "rgb":
        raise ValueError(
            "Unsupported FEATURE_COLOR_MODE. Use 'rgb' or 'grayscale'."
        )

    return resized.reshape(-1)


def get_expected_feature_count() -> int | None:
    """Read the expected feature count from the fitted scaler when available."""
    try:
        scaler = load_scaler()
    except Exception:
        return None

    return getattr(scaler, "n_features_in_", None)


def align_feature_count(features, expected_count: int | None):
    """Trim or pad features so they match the model's expected width."""
    if expected_count is None or features.shape[1] == expected_count:
        return features

    if features.shape[1] > expected_count:
        return features[:, :expected_count]

    padding = np.zeros(
        (features.shape[0], expected_count - features.shape[1]),
        dtype=features.dtype,
    )
    return np.hstack([features, padding])


def get_joblib_inference_unavailable_reason() -> str | None:
    """Explain why the sklearn joblib inference API cannot run."""
    if np is None or cv2 is None:
        return "NumPy or OpenCV is not installed in this environment."
    if not os.path.isfile(MODEL_BUNDLE_PATH):
        return f"Model bundle not found at {MODEL_BUNDLE_PATH}."
    if not os.path.isfile(SCALER_PATH):
        return f"Scaler not found at {SCALER_PATH}."
    if not os.path.isfile(LABEL_ENCODER_PATH):
        return f"Label encoder not found at {LABEL_ENCODER_PATH}."
    if USE_PCA and not os.path.isfile(PCA_PATH):
        return f"PCA artifact not found at {PCA_PATH}."

    try:
        load_models()
        load_scaler()
        load_label_encoder()
        if USE_PCA:
            load_pca()
    except Exception as exc:
        return describe_joblib_loading_error(exc)

    return None


def infer_image(filepath, model_name="KNN", model_path=MODEL_BUNDLE_PATH):
    """Run inference for one image using a selected sklearn model."""
    unavailable_reason = get_joblib_inference_unavailable_reason()
    if unavailable_reason is not None:
        raise RuntimeError(unavailable_reason)

    models_dict = load_models(model_path)

    if model_name not in models_dict:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(models_dict.keys())}"
        )

    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filepath}")

    feat = extract_features(img).astype(np.float32).reshape(1, -1)

    scaler = load_scaler()
    expected_feature_count = getattr(scaler, "n_features_in_", None)
    actual_feature_count = feat.shape[1]
    if (
        expected_feature_count is not None
        and actual_feature_count != expected_feature_count
    ):
        raise ValueError(
            "Feature mismatch between inference and training pipeline. "
            f"Extractor produced {actual_feature_count} features, "
            f"but scaler expects {expected_feature_count}. "
            f"Current FEATURE_IMAGE_SIZE={FEATURE_IMAGE_SIZE} and "
            f"FEATURE_COLOR_MODE='{FEATURE_COLOR_MODE}'."
        )

    feat = scaler.transform(feat)

    if USE_PCA:
        pca = load_pca()
        feat = pca.transform(feat)

    le = load_label_encoder()
    model = models_dict[model_name]
    model_expected_feature_count = getattr(model, "n_features_in_", None)
    feat = align_feature_count(feat, model_expected_feature_count)
    pred_idx = model.predict(feat)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(feat)[0].tolist()

    return {
        "model": model_name,
        "prediction": pred_label,
        "prediction_index": int(pred_idx),
        "probabilities": prob,
    }


def get_inference_unavailable_reason() -> str | None:
    """Explain why inference is unavailable in the current environment."""
    if torch is None:
        return "PyTorch is not installed in this deployment environment."
    if np is None or cv2 is None:
        return "Image processing dependencies are not installed in this deployment environment."

    weights_path = os.environ.get(
        "MODEL_WEIGHTS",
        os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    )
    if not os.path.isfile(weights_path):
        return f"Model weights were not found at {weights_path}."

    return None


def load_model() -> "torch.nn.Module | None":
    """Load the YOLOv5 model if PyTorch is available.

    Returns ``None`` if the model cannot be loaded for any reason.  When
    loading the model the code uses ``torch.hub.load`` to fetch the
    repository ``ultralytics/yolov5`` and a custom model specified by
    ``MODEL_WEIGHTS`` environment variable or a default path.  See the
    project README for details on obtaining the weights file.  Using
    ``trust_repo=True`` (available from PyTorch 1.13 onward) prevents
    warnings about untrusted repositories.
    """
    unavailable_reason = get_inference_unavailable_reason()
    if unavailable_reason is not None:
        print(f"Model loading skipped. {unavailable_reason}")
        return None

    weights_path = os.environ.get(
        "MODEL_WEIGHTS",
        os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    )

    try:
        # The 'custom' argument tells YOLOv5 to load a user‑trained model
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=weights_path, trust_repo=True
        )
        # Disable gradient computation for inference
        model.eval()
        return model
    except Exception as exc:
        print(f"Could not load YOLO model: {exc}")
        return None


def get_results_dir() -> str:
    """Return a writable directory for generated outputs."""
    results_dir = os.path.join(tempfile.gettempdir(), "helmet_detection_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def create_app() -> Flask:
    """Factory function to create and configure the Flask application."""
    app = Flask(__name__)

    # Load the model once when the app starts.  If loading fails the variable
    # will be ``None`` and prediction will be skipped.
    model = load_model()

    @app.route("/")
    def index() -> str:
        """Render the upload form.  Displays the last result if available."""
        return render_template("index.html")

    @app.route("/api/health")
    def api_health():
        """Return basic API health and artifact readiness."""
        joblib_reason = get_joblib_inference_unavailable_reason()
        available_models = []
        if joblib_reason is None:
            try:
                available_models = list(load_models().keys())
            except Exception as exc:
                joblib_reason = describe_joblib_loading_error(exc)

        return jsonify(
            {
                "status": "ok",
                "joblib_inference_ready": joblib_reason is None,
                "joblib_inference_error": joblib_reason,
                "expected_feature_count": get_expected_feature_count(),
                "feature_image_size": FEATURE_IMAGE_SIZE,
                "feature_color_mode": FEATURE_COLOR_MODE,
                "available_models": available_models,
            }
        )

    @app.route("/api/infer", methods=["POST"])
    def api_infer():
        """Run sklearn joblib inference for an uploaded image."""
        upload = request.files.get("file")
        model_name = request.form.get("model_name", "KNN")

        if not upload or upload.filename == "":
            return jsonify({"error": "Please upload an image file with field name 'file'."}), 400

        unavailable_reason = get_joblib_inference_unavailable_reason()
        if unavailable_reason is not None:
            return jsonify({"error": unavailable_reason}), 503

        temp_path = None
        try:
            suffix = os.path.splitext(secure_filename(upload.filename) or "upload.jpg")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".jpg") as temp_file:
                upload.save(temp_file)
                temp_path = temp_file.name

            result = infer_image(temp_path, model_name=model_name)
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            return jsonify({"error": f"Inference failed: {exc}"}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Handle image upload and run inference to detect helmets.

        The endpoint expects a file input named ``file``.  If a model is
        loaded, it will annotate the image and save it into the results
        directory.  The annotated image's relative path is passed back to
        the template for display.  Errors (e.g., no file uploaded) are
        returned to the user via the template.
        """
        upload = request.files.get("file")
        if not upload or upload.filename == "":
            return render_template(
                "index.html", error="Please select an image to upload."
            )

        unavailable_reason = get_inference_unavailable_reason()
        if model is None:
            return render_template(
                "index.html",
                error=(
                    "Helmet detection is unavailable in this deployment. "
                    f"{unavailable_reason or 'The model could not be loaded.'}"
                ),
            )

        # Read the uploaded image into a NumPy array
        file_bytes = np.fromfile(upload, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return render_template(
                "index.html",
                error="The uploaded file could not be processed as an image.",
            )

        annotated_path = None
        if model is not None:
            results_dir = get_results_dir()
            # Perform inference; results.render() returns a list of annotated
            # images corresponding to the input batch.  Here we upload a
            # single image, so take the first result.
            results = model(img)
            annotated_image = results.render()[0]  # type: ignore[index]
            # Construct a filename based on the original name
            name, _ = os.path.splitext(upload.filename)
            filename = f"{name}_result.jpg"
            annotated_path = os.path.join(results_dir, filename)
            # Save the image in BGR format expected by OpenCV
            cv2.imwrite(annotated_path, annotated_image)

        # Build a relative URL for the saved image so that it can be served
        relative_image = (
            url_for("static", filename=f"results/{os.path.basename(annotated_path)}")
            if annotated_path
            else None
        )
        return render_template("index.html", image=relative_image)

    return app


app = create_app()
application = app


if __name__ == "__main__":
    # When run directly, create the app and run the development server.  The
    # host is set to 0.0.0.0 to allow access via Docker or remote hosts.
    app.run(host="0.0.0.0", port=5000, debug=True)
