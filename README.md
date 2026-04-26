# ODOL Truck Detection AI & Web App 🚛

**Live Web Demo:** [https://huggingface.co/spaces/Kaiichin/odol-truck-detection-website](https://huggingface.co/spaces/Kaiichin/odol-truck-detection-website)

## Project Overview
This project is an artificial intelligence pipeline designed to detect Over Dimension Overload (ODOL) trucks. The system uses a two-stage computer vision approach to first locate a truck in an image, and then classify whether that specific truck is normal or overloaded. The AI is wrapped in a custom FastAPI backend and served through a modern, glassmorphism-styled web interface.

## The AI Pipeline Process
1. **Detection (YOLOv8):** The first stage takes an uploaded image and passes it through a custom-trained YOLOv8 model. The model scans the image to find the exact bounding box of the truck.
2. **Classification (MobileNetV2):** The truck is cropped out using the bounding box from stage 1 and passed into a custom MobileNetV2 classifier. This brain analyzes the truck's physical dimensions and outputs a prediction of either `NORMAL` or `ODOL`.
3. **Web Application (FastAPI + JS):** The backend merges the AI results, draws a colored bounding box on the original image (Green for Normal, Red for ODOL), and sends it back to the Vanilla JavaScript frontend using Base64 encoding.

## Repository Structure
- **`Truck Detection YOLO Model/`**: Contains the Colab notebooks used to train the YOLOv8 object detector.
- **`MobileNet Training ODOL/`**: Contains the scripts and notebooks used to train the MobileNetV2 classifier.
- **`ODOL Truck Detection Model Final/`**: Contains the pipeline tests linking both AI models together locally.
- **`ODOL Truck Detection Web/`**: The complete, deployable web application source code (HTML/CSS/JS Frontend + FastAPI Backend + Dockerfile).

*Note: The heavy `.pt` and `.h5` model weights are hosted separately on the Hugging Face Hub to keep this repository lightweight and fast.*

## Current Limitations
During early experiments, the AI was tested against real-time top-down CCTV footage. Because the dataset used to train the classifier consists mostly of **side-view** images, the accuracy heavily fluctuates when viewing trucks from top-down angles or in video motion. Currently, this system is optimized strictly for static, side-profile images of trucks and is not yet ready for live CCTV video implementation.
