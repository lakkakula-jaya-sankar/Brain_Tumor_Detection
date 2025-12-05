🧠 Brain Tumor Detection & Medicine Suggestion System

A Machine Learning + Flask Web Application

This project is an end-to-end Brain Tumor Detection System that analyzes MRI brain scans using a CNN deep learning model, detects tumors, estimates approximate tumor size, suggests medicine, and even locates the nearest hospital using geolocation and the OpenStreetMap API.
It also includes User Authentication (Login/Register) and Email Notifications for prediction results.

🚀 Features
🧑‍⚕️ 1. Brain Tumor Classification

Uses Convolutional Neural Network (CNN) to classify MRI scans as:
✔️ No Tumor Detected
✔️ Tumor Detected

📏 2. Tumor Size Estimation

Uses contour extraction and pixel-to-cm calculation.

Displays approximate tumor size in cm.

💊 3. Medicine Suggestion System

Depending on tumor severity:

Tumor Size	Suggested Medicine
≤ 3 cm	Aspirin
3–4.5 cm	Dexamethasone
4.5–5.5 cm	Temozolomide
> 5.5 cm	Go to Hospital
🏥 4. Nearest Hospital Locator

Uses OpenStreetMap Overpass API

Gets user latitude + longitude

Returns nearest hospital name & location.

📩 5. Email Notification

After prediction, user receives mail containing:

Prediction

Tumor Size

Medicine / Hospital details

🔐 6. User Authentication

Register

Login

Secure Password Hashing

Session Handling

🛠️ Tech Stack
Backend

Python

Flask

SQLite

OpenCV

NumPy

Requests

smtplib (Email)

Machine Learning

CNN using Keras

TensorFlow

Numpy

Train/Test Split

Accuracy Score

Front-end

HTML

CSS

JavaScript

📁 Project Structure
project/
│
├── app.py
├── Model/
│   ├── model.json
│   ├── model_weights.h5
│   ├── history.pckl
│   ├── myimg_data.txt.npy
│   └── myimg_label.txt.npy
│
├── dataset/
│   ├── no/
│   └── yes/
│
├── templates/
│   ├── index.html
│   ├── login.html
│   └── register.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── img/
│
└── README.md
