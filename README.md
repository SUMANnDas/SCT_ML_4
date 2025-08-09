This project is part of my SkillCraft Internship, Task 4, where I developed and integrated the backend system for an AI-powered Gesture Recognition application. The backend is built using Flask and integrates a trained deep learning model to process uploaded gesture images and return predictions in real-time.

🚀 Features
AI Model Training:

Prepared dataset for gesture classification

Trained a deep learning model for accurate predictions

Optimized performance for real-time inference

Backend API:

Flask REST API endpoints for secure file uploads

Validates input files (image formats)

Preprocesses input before passing it to the model

Returns prediction results in JSON format

Integration Ready:

Designed to work seamlessly with the frontend

Scalable structure for future enhancements

🛠️ Tech Stack
Backend Framework: Flask (Python)

AI/ML: TensorFlow / Keras / OpenCV (depending on model)

Data Handling: NumPy, Pillow

API: RESTful API endpoints for prediction

📌 How It Works
User uploads an image of a hand gesture through the frontend.

The backend receives the file via /predict endpoint.

The image is validated and preprocessed.

The trained gesture recognition model processes the input.

The backend sends the prediction result back to the frontend.

🔹 Key Learnings from Task 4
Designing REST APIs for AI model integration

Building scalable backend systems with Flask

Handling file uploads securely

Deploying trained ML models in production

Creating an end-to-end AI application workflow
