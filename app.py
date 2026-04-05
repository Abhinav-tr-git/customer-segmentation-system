import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from loguru import logger

from src.pipelines.inference_pipeline import run_inference
from src.utils.logging import setup_logging

# Initialise logging at module level so it runs regardless of how the app is
# started (python app.py, gunicorn, pytest, etc.).
setup_logging("config/config.yaml")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"csv"}
# Reject uploads larger than 50 MB — prevents accidental / malicious DoS.
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Ensure the upload directory exists at startup.
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --- Validate the upload ---
    if "file" not in request.files:
        return render_template("index.html", error="No file part in the request.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not _allowed_file(file.filename):
        return render_template("index.html", error="Only CSV files are accepted.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        results = run_inference(filepath)

        # Normalise cluster labels to str so Jinja2 and JS both get consistent
        # string keys — avoids subtle int-vs-string key mismatch in templates.
        results["cluster"] = results["cluster"].astype(str)

        summary = (
            results.groupby("cluster")
            .agg(recency=("recency", "mean"), frequency=("frequency", "mean"), monetary=("monetary", "mean"))
            .round(2)
            .to_dict("index")
        )
        cluster_counts = results["cluster"].value_counts().to_dict()

        output_filename = "results_" + filename
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        results.to_csv(output_path, index=True)

        return render_template(
            "results.html",
            summary=summary,
            counts=cluster_counts,
            download_link=output_filename,
        )

    except FileNotFoundError as e:
        # Model artifacts missing — user needs to train first.
        logger.error(f"Artifact not found: {e}")
        return render_template("index.html", error=str(e))

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render_template("index.html", error=f"An unexpected error occurred: {e}")


@app.route("/download/<filename>")
def download_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
