# ODOL Truck Screening System

A two-stage computer-vision system for screening trucks as **ODOL** or **NOT ODOL**:

1. **YOLOv8m** detects trucks and creates truck crops.
2. **MobileNetV2** classifies each crop as `ODOL` or `NOT ODOL`.

> This is a visual screening system. It does not directly measure vehicle weight or physical dimensions.

## Open in Google Colab

| Notebook | Purpose | Colab |
|---|---|---|
| YOLOv8m Truck Detection | Clean grouped dataset, train detector, evaluate test set | [Open](https://colab.research.google.com/drive/1nKlu7S08mjkCd_IDzjOKZxR13hph3wfu?usp=sharing) |
| MobileNetV2 ODOL Classification | Create truck crops, train classifier, evaluate test set | [Open](https://colab.research.google.com/drive/1Z54vTQMAd9XROAza_Hm4fX_5fXsqL37G?usp=sharing) |
| Image and Video Pipeline | Run both models on uploaded images or MP4 videos | [Open](https://colab.research.google.com/drive/1sxt-qZ5TXp77SrZuVLe6p-k1_AP6C5M_?usp=sharing) |

## Latest results

### YOLOv8m truck detector

| Metric | Result |
|---|---:|
| Precision | 0.9618 |
| Recall | 0.8725 |
| F1 score | 0.9150 |
| mAP50 | 0.9386 |
| mAP50–95 | 0.8222 |

The YOLO dataset contains 1,769 annotated images. Original and augmented images are grouped by removing `.rf.HASH` before the 70/15/15 split to prevent data leakage.

### MobileNetV2 ODOL classifier

| Metric | Result |
|---|---:|
| Accuracy | 0.9056 |
| Precision | 0.8917 |
| Recall | 0.9032 |
| F1 score | 0.8974 |

Confusion matrix:

```text
[[167  17]
 [ 15 140]]
```

The held-out MobileNetV2 test set contains 339 crops, so one prediction changes accuracy by about 0.30 percentage points.

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
MobilenetV2/
└── MobileNetV2_Truck_Detection_(new_code).ipynb
ODOL Pipeline/
└── ODOL_Pipeline_NewCode.ipynb
Weights/
├── best.pt
└── mobilenet_odol_classifier(V2).keras
YOLOv8m/
└── YOLO_Truck_Detection_(new_code).ipynb
```

Datasets, generated runs, output videos, and `last.pt` are not committed. The model weights listed above are included in this repository.

## Model weights

The pipeline loads these model files, which are available in the [`Weights/`](Weights/) folder. When running the pipeline notebook in Colab, upload both files when prompted:

- `best.pt`
- `mobilenet_odol_classifier(V2).keras`

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
