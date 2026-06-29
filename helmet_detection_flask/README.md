# Helmet Detection Flask App

This repository contains a minimal Flask web application for detecting helmets in images using a pre‑trained object detection model. It provides a simple interface that allows users to upload an image and receive an annotated result showing the detected helmets.

## Features

* **Flask Interface:** A clean web interface built with [Flask](https://palletsprojects.com/p/flask/) for uploading images and displaying results.
* **YOLOv5 Integration:** The app uses PyTorch to perform object detection when the ML dependencies and model weights are available.
* **Automatic Results Storage:** Annotated images are saved in `static/results/` and served back through the web app for easy viewing.

## Setup

1. **Clone or download this repository** into your environment.
2. **Install dependencies** using pip. A recommended way is to create a virtual environment first:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements-ml.txt
   ```

3. **Provide your model weights**.  Place your custom YOLOv5 weights file (for example, `best.pt`) in a directory named `weights` at the project root (the same directory as `app.py`).  The application tries to load `weights/best.pt` by default.  Alternatively, set the `MODEL_WEIGHTS` environment variable to the full path of your weights file.

4. **Run the application** in development mode:

   ```bash
   python app.py
   ```

   The app will start a development server on `http://0.0.0.0:5000`.  Open this address in your web browser and follow the on‑screen instructions to upload an image.

## Deployment notes

* `requirements.txt` is intentionally minimal for serverless deployment and only installs Flask.
* `requirements-ml.txt` contains the heavy ML dependencies needed for local inference or deployment on a VM/container platform.
* Vercel Python functions run on AWS Lambda and are not a good fit for bundling PyTorch and OpenCV together with model weights. On Vercel, this app will deploy, but prediction will show a message explaining that inference is unavailable unless you move the ML inference to another service.

## Notes

* If PyTorch, image-processing libraries, or model weights are missing, the app will still run but will not annotate images.
* The YOLOv5 model weights must be compatible with the number of classes in your training dataset (e.g., helmet vs. no helmet). You can train your own model and then copy the resulting weights into this project for local or container-based deployment.

## Citation

The general structure of this application is inspired by an open source example of deploying a helmet detection model with Flask and YOLOv5.  That project loads a custom model using `torch.hub.load` and renders prediction results onto the input image before returning it via a Flask route【767599868399950†L250-L276】.
