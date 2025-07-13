# 😊 Real-Time Emotion Detection using MobileNetV2 + OpenCV

This project uses a fine-tuned MobileNetV2 model to detect human facial emotions in **images**, **videos**, and **live webcam** streams. It applies deep learning for expression classification and OpenCV for face detection and visual overlay.

---

## 🧠 Emotions Detected

- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😄
- Sad 😢
- Surprise 😲
- Neutral 😐

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** – For building and training the MobileNetV2 model  
- **OpenCV** – For real-time face detection and visualization  
- **NumPy & Matplotlib** – Data processing and visualization  
- **FER2013 Dataset** – Standard dataset for training emotion recognition

---

## 📦 Folder Structure

\`\`\`
Emotion-Detection/
│
├── final_emotion_model.h5              # Trained MobileNetV2 model
├── Emotion_Prediction_MobileNetV2.ipynb # Notebook for prediction & visualization
├── detectedimg.jpg                     # Sample output (auto-generated)
├── README.md                           # This file
└── dataset/
    ├── train/
    └── test/
\`\`\`

---

## 🚀 Features

### ✅ Emotion Prediction from Image
Detects all faces in an image and overlays bounding boxes and predicted emotions with confidence.

\`\`\`python
predict_emotion_from_path("path_to_image.jpg")
\`\`\`

---

### ✅ Live Webcam Emotion Detection *(Updated Feature)*  
Predicts **Top-3 emotions per face with confidence %** in real time, using webcam input.

\`\`\`python
run_webcam_emotion_detection()
\`\`\`

---

### ✅ Video File Emotion Detection
Processes every frame in a video file, detects faces, and predicts emotion with overlay.

\`\`\`python
run_video_emotion_detection("path_to_video.mp4")
\`\`\`

---

### ✅ Sample Output of Expression Prediction

Once a face is detected and emotion is predicted, a sample image (`detectedimg.jpg`) is saved automatically showing bounding box and emotion label.

---

## 📷 Visualization

- Bounding boxes drawn around faces.
- Emotion labels displayed above each face.
- Top-3 emotions (for webcam) with confidence %.
- Emotion-specific color codes:
  - Happy → Green
  - Angry → Red
  - Sad → Blue
  - and so on...

---

## 💾 Prediction Sample

### 🖼️ Images

<p align="center">
  <img src="Images/structure_gan_ti.png" alt="Optimized GaN-Ti Nanotube Structure" width="500"/>
</p>

### 🎥 Videos

<p align="center">
  <img src="Images/band_structure.png" alt="Band Structure Plot" width="500"/>
</p>

### 📷 Webcam

<p align="center">
  <img src="Images/dos_plot.png" alt="Density of States" width="500"/>
</p>


---

## ⚠️ Note on Accuracy

📉 **Current model accuracy is ~40%**, which means predictions may not always be reliable.  
Improving the model’s performance is part of future work, including better data preprocessing, augmentation, and fine-tuning.

---

## 🧪 Model Evaluation

To evaluate accuracy on the test set:

\`\`\`python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/test', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
\`\`\`

---

## 🔧 Setup Instructions

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/Emotion-Detection.git
cd Emotion-Detection
```

2. **Install dependencies:**

```bash
pip install tensorflow==2.10.1 opencv-python numpy matplotlib
```

3. **Run the notebook:**

```bash
jupyter notebook Emotion_Prediction_MobileNetV2.ipynb
```

---

## 💡 Future Enhancements

| Feature                                | Description |
|----------------------------------------|-------------|
| 🌐 Web Deployment (Streamlit/Gradio)   | Deploy model with UI for upload/live webcam |
| 📈 Accuracy Visualizations              | Show confusion matrix, F1 score, precision |
| 🎥 Save Processed Videos                | Export annotated video with timestamped emotions |
| 📦 Real-time Logging                    | Store results in CSV or database |
| 🧑‍🤝‍🧑 Group Face Emotion Analysis       | Show crowd emotion summary |
| 🗣️ Audio + Emotion Fusion               | Combine voice tone and face for better inference |
| 📱 Mobile App Integration               | Deploy using TensorFlow Lite on mobile |
| 🔁 Model Retraining                     | Improve model performance with more diverse datasets |
| 🤖 Use Better Architecture              | Try EfficientNet or Vision Transformers for higher accuracy |

---

## 📝 License

This project is licensed under the MIT License.

---


## 🙌 Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- TensorFlow and OpenCV communities
""")

## 📬 Contact

👤 Kirti Vardhan Singh  
📧 Email: kirtivardhan7549@gmail.com  
🏫 Department of Computer Science and Engineering  
Centurion University of Technology and Management, Bhubaneswar, India

---

<div align="center">
  Made with ❤️ for materials simulation and nanoscience.
</div>
```
