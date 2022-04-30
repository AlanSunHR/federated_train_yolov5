# Evaluation of Federated Training Object Detection Neural Networks

## Pre-requests
1. Git clone the repository
```bash
git clone https://github.com/AlanSunAlan/federated_train_yolov5.git
```
2. cd to the cloned repository
```bash
cd federated_train_yolov5
```
3. Git clone the yolov5 repository
```bash
git clone https://github.com/ultralytics/yolov5.git
```
4. Install the pre-requested python packages as said in the yolov5 repository
```bash
cd yolov5
pip install -r requirements.txt
```

## How to run centralized training:
1. Run cnetralized training of yolov5s on VisDrone Dataset:
```bash
python3 centralized_training.py
```
### Note: This will download the VisDrone Dataset in folder "datasets" on root directory of this repository

## How to run federated training:
1. Run the centralized training script before runing federated training.
2. Split the dataset into several sub-datasets
```bash
python3 split_data.py
```
3. Run the federated training on evenly distributed datasets:
```bash
python3 federated_train_even.py
```
4. Run the federated training on non-iid sub-datasets:
```bash
python3 federated_train_noniid.py
```
