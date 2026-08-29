# ODOL Truck Screening System

A two-stage computer-vision system for screening trucks as **ODOL** or **NOT ODOL**:

1. **YOLOv8m** detects trucks and creates truck crops.
2. **MobileNetV2** classifies each crop as `ODOL` or `NOT ODOL`.

> This is a visual screening system. It does not directly measure vehicle weight or physical dimensions.

## Open in Google Colab

| Notebook | Purpose | Colab |
|---|---|---|
| YOLOv8m Truck Detection | Clean grouped dataset, train detector, evaluate test set | [Open](https://colab.research.google.com/github/kaiichin/ODOL-Truck-Detection/blob/main/notebooks/01_YOLOv8m_Truck_Detection.ipynb) |
| MobileNetV2 ODOL Classification | Create truck crops, train classifier, evaluate test set | [Open](https://colab.research.google.com/github/kaiichin/ODOL-Truck-Detection/blob/main/notebooks/02_MobileNetV2_ODOL_Classification.ipynb) |
| Image and Video Pipeline | Run both models on uploaded images or MP4 videos | [Open](https://colab.research.google.com/github/kaiichin/ODOL-Truck-Detection/blob/main/notebooks/03_ODOL_Image_Video_Pipeline.ipynb) |

## Latest results

### YOLOv8m truck detector

| Metric | Result |
|---|---:|
| Precision | 0.9481 |
| Recall | 0.8700 |
| F1 score | 0.9074 |
| mAP50 | 0.9363 |
| mAP50–95 | 0.8144 |

The YOLO dataset contains 1,769 annotated images. Original and augmented images are grouped by removing `.rf.HASH` before the 70/15/15 split to prevent data leakage.

### MobileNetV2 ODOL classifier

| Metric | Result |
|---|---:|
| Accuracy | 0.9024 |
| Precision | 0.8889 |
| Recall | 0.9600 |
| F1 score | 0.9231 |

Confusion matrix:

```text
[[13  3]
 [ 1 24]]
```

The held-out MobileNetV2 test set contains only 41 crops, so one prediction changes accuracy by about 2.44 percentage points.

## Image and video pipeline

```text
Image or video frame
        ↓
YOLOv8m detects each truck
        ↓
Detected truck is cropped
        ↓
MobileNetV2 classifies the crop
        ↓
Annotated image or video is saved
```

Video mode includes:

- ByteTrack IDs for trucks across frames
- Five-frame MobileNetV2 prediction averaging
- `UNCERTAIN` output for ODOL scores between 40% and 60%
- `TOO FAR` output for crops smaller than 80 × 80 pixels
- Browser-compatible MP4 output

## Repository structure

```text
notebooks/
├── 01_YOLOv8m_Truck_Detection.ipynb
├── 02_MobileNetV2_ODOL_Classification.ipynb
└── 03_ODOL_Image_Video_Pipeline.ipynb
```

Datasets, generated runs, output videos, `last.pt`, and model weights are not committed.

## Model weights

The pipeline downloads these assets from the [`v2.0.0` GitHub Release](https://github.com/kaiichin/ODOL-Truck-Detection/releases/tag/v2.0.0):

- `best.pt`
- `mobilenet_odol_classifier.keras`

## Datasets

- **YOLO:** Kaggle Trucks Detection, `Truck counting.v1i.yolov8`
- **MobileNetV2:** Roboflow Truck Over Dimension, version 2

The datasets are downloaded by the training notebooks and are not stored in this repository.

For the Roboflow notebook, add a Google Colab Secret named:

```text
ROBOFLOW_API_KEY
```

## Limitations

- The models are sensitive to camera angle, distance, lighting, and differences from the training datasets.
- Small distant trucks can be missed by YOLO or marked `TOO FAR`.
- MobileNetV2 predictions can be uncertain on unseen CCTV angles.
- Tracking and frame averaging reduce label flicker but do not guarantee a correct vehicle-level decision.
- The system must not be presented as a replacement for legal weight or dimension measurement.
