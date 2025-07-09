# Classification and Captioning Aircraft Damage

### Classify aircraft damage using a pre-trained VGG16 model and generate captions using a Transformer-based pretrained model.

This project aims to automate the classification of aircraft damage into two categories: "dent" and "crack." 
For this, we will utilize feature extraction with a pre-trained VGG16 model to classify the damage from aircraft images. 
Additionally, we will use a pre-trained Transformer model to generate captions and summaries for the images.

### Final Output

A trained model capable of classifying aircraft images into "dent" and "crack" categories, enabling automated aircraft damage detection.
A Transformer-based model that generates captions and summaries of images.

You will be using the [Aircraft dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar). 
The dataset is taken from the Original Source: [Roboflow Aircraft Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) 
Provided by a Roboflow user, License: CC BY 4.

<h2>Table of Contents</h2>
 
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <p><font size="5">Part 1 - Classification Problem: Classifying the defect on the aircraft as 'dent' or 'crack'</p>
<font size="3">
        1. <a href="#1.1-Dataset-Preparation">1.1 Dataset Preparation</a><br>
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


# 1.1 Dataset-Preparation
