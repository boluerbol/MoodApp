# MoodApp
MoodApp is a mobile application that detects human emotions (happy, sad, surprised) from images. Built using a pre-trained Convolutional Neural Network (CNN) model and deployed as an Android app, this project provides users with a simple interface to take a picture and receive an immediate mood prediction.

Features
Emotion Detection: Recognizes three primary emotions: happy, sad, and surprised.
User-Friendly Interface: Allows users to capture images and view predictions instantly.
Mobile Deployment: Converted the model to TensorFlow Lite for efficient mobile processing.
Cross-Platform Development: Uses Kivy and Buildozer to package the Python-based app for Android.


Project Structure
bash

MoodApp/
├── assets/              # Assets like app icons
├── model/               # Saved model files and TensorFlow Lite model (.tflite)
├── app/                 # Kivy app interface scripts
│   ├── main.py          # Main app logic
│   ├── ui.kv            
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation



Setup and Installation
Prerequisites
-Python 3.7+
-Virtual environment (recommended)
-TensorFlow and Kivy dependencies (see requirements.txt)
-Android SDK and NDK
-Buildozer

# Step 1: Clone the Repository

git clone https://github.com/yourusername/MoodApp.git
cd MoodApp

# Step 2: Set Up the Environment

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Step 3: Convert Model to TensorFlow Lite

import tensorflow as tf
# Load and convert model
model = tf.keras.models.load_model("./model/model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Step 4: Build APK with Buildozer
Ensure all dependencies are installed and then run Buildozer to generate the APK file.

buildozer -v android debug

# Step 5: Run on an Android Device
Once the APK is generated, transfer it to an Android device to install and test.

Usage
Launch the app and use the camera button to capture an image.
Click "Predict" to get the mood prediction.
View the mood displayed on the screen.

Contributing
Fork the repository.
Create a new branch with your feature or bug fix.
Push to your branch and create a pull request.

 
