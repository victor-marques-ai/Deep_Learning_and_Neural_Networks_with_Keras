# Classification and Captioning Aircraft Damage

### Classify aircraft damage using a pre-trained VGG16 model and generate captions using a Transformer-based pretrained model.

This project aims to automate the classification of aircraft damage into two categories: "dent" and "crack." 
For this, we will utilize feature extraction with a pre-trained VGG16 model to classify the damage from aircraft images. 
Additionally, we will use a pre-trained Transformer model to generate captions and summaries for the images.

### Final Output

A trained model capable of classifying aircraft images into "dent" and "crack" categories, enabling automated aircraft damage detection.
A Transformer-based model that generates captions and summaries of images.

You will be using the [Aircraft dataset (16.4 MB)](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar). 
The dataset is taken from the Original Source: [Roboflow Aircraft Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) 
Provided by a Roboflow user, License: CC BY 4.

## Table of Contents

[1.1 Dataset Preparation](#11-dataset-preparation)
 
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <p><font size="5">Part 1 - Classification Problem: Classifying the defect on the aircraft as 'dent' or 'crack'</p>
<font size="3">
        <br>
        2. <a href="#1.2-Data-Preprocessing">1.2 Data Preprocessing</a><br>
        3. <a href="#1.3-Model-Definition">1.3 Model Definition</a><br>
        4. <a href="#1.4-Model-Training">1.4 Model Training</a><br>
        5. <a href="#1.5-Visualizing-Training-Results">1.5 Visualizing Training Results</a><br>
        6. <a href="#1.6-Model-Evaluation">1.6 Model Evaluation</a><br>
        7. <a href="#1.7-Visualizing-Predictions">1.7 Visualizing Predictions</a><br>
    <br>
<p><font size="5">Part 2: Image Captioning and Summarization using BLIP Pretrained Model</p>
<font size="3">
        1. <a href="#2.1-Loading-BLIP-Model">2.1 Loading BLIP Model</a><br>
        2. <a href="#2.2-Generating-Captions-and-Summaries">2.2 Generating Captions and Summaries</a><br>
        <br>

### Task List
To achieve the above objectives, you will complete the following tasks:

- Task 1: Create a `valid_generator` using the `valid_datagen` object
- Task 2: Create a `test_generator` using the `test_datagen` object
- Task 3: Load the VGG16 model
- Task 4: Compile the model
- Task 5: Train the model
- Task 6: Plot accuracy curves for training and validation sets 
- Task 7: Visualizing the results 
- Task 8: Implement a Helper Function to Use the Custom Keras Layer
- Task 9: Generate a caption for an image using the using BLIP pretrained model
- Task 10: Generate a summary of an image using BLIP pretrained model


### Required Libraries
```
!pip install pandas==2.2.3
!pip install tensorflow==2.17.1
!pip install pillow==11.1.0
!pip install matplotlib==3.9.2
!pip install transformers==4.38.2
!pip install torch
```

> Suppress the tensorflow warning messages</br>
>The following code to suppress the warning messages due to use of CPU architechture for tensoflow.
> 
```
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### Import Libraries
```
import zipfile
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import random
```

### Set seed
```
# Set seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
```
### Pretrained Models
>### **[ResNet](https://keras.io/api/applications/resnet/), [VGG](https://keras.io/api/applications/vgg/) (Image Classification)**:
>These are pretrained models commonly used for image classification tasks. They have learned from millions of images and can be fine-tuned for specific image-related tasks.

>### **BLIP (Image Captioning and Summarization)**:
>BLIP is a pretrained model that can generate captions and summaries for images. It has already been trained on image-text pairs, so it can easily generate descriptive captions for new images.


# 1.1-Dataset-Preparation



# 1.2-Data-Preprocessing
