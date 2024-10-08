
# Medical Image Segmentation and Classification for MICCAI IUGC 2024: Intrapartum Ultrasound Grand Challenge

This repository contains code for the classification and segmentation of medical scanned images of babies inside mothers. The workflow consists of several scripts to train models, preprocess data, and make predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training the Segmentation Model](#training-the-segmentation-model)
  - [Preprocessing Data](#preprocessing-data)
  - [Training the Classification Model](#training-the-classification-model)
  - [Using Trained Models](#using-trained-models)
- [Examples](#examples)
- [Results](#results)
- [References](#references)

## Installation

Make sure you have Python 3.x installed. You may also need to install the following dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Segmentation Model

1. Start by running the `Train2.py` script to train the segmentation model. This will save the model in `.pth` format.

   ```bash
   python Train2.py
   ```

### Preprocessing Data

2. Next, run the `pre_train.py` script to convert `.avi` files into `.png` format. This format will be used for training the classification model.

   ```bash
   python pre_train.py
   ```

### Training the Classification Model

3. After preprocessing, run the `Train1.py` script to train the classification model.

   ```bash
   python Train1.py
   ```

### Using Trained Models

4. You can initialize the models and convert `.pth` files to `.pickle` format using the `modelInter` class. Alternatively, you can skip the training process and immediately load the trained models:

   ```python
   from modelInter import ModelInter

   MDL = ModelInter()
   MDL.load()
   ```

5. To make predictions, use the `predict` method, where `X` is a single image of shape `(3, 512, 512)` in the form of a NumPy array:

   ```python
   result = MDL.predict(X)
   ```

## Examples

The `Test.py` script demonstrates how to use the models and visualize the results. 

```bash
python Test.py
```

## Results

The code will create `segmentation_result.png` in the `Task3` folder, which can be used for further measurements. HSD and AoP angles can be measured by running:

```bash
python RunHere.py
```

## References

- [Least Squares Ellipse Fitting](https://github.com/bdhammel/least-squares-ellipse-fitting)
- [IUGC 2024 Starting Tool Kit](https://github.com/maskoffs/IUGC2024/tree/main/starting_tool_kit)
- [Segmentation Models for PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [MICCAI IUGC 2024: Intrapartum Ultrasound Grand Challenge link](https://codalab.lisn.upsaclay.fr/competitions/18413#results)
```