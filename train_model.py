# Hand Gesture Recognition System
# Complete end-to-end implementation

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
import shutil
from pathlib import Path

class HandGestureRecognizer:
    def __init__(self, img_height=128, img_width=128, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.class_names = [
            '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
            '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
        ]
        self.num_classes = len(self.class_names)
        
    def check_dataset_structure(self, dataset_path):
        """Check and fix dataset structure if needed"""
        print("Checking dataset structure...")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found!")
        
        # Check if it's subject-based structure (00, 01, 02, ...)
        subject_folders = [d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()]
        
        # Check if it's direct gesture structure (01_palm, 02_l, ...)
        gesture_folders = [d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d)) and d in self.class_names]
        
        total_images = 0
        
        if subject_folders:
            print(f"✅ Found subject-based structure with {len(subject_folders)} subjects")
            # Count images in subject structure
            for subject in subject_folders:
                for gesture_class in self.class_names:
                    gesture_path = os.path.join(dataset_path, subject, gesture_class)
                    if os.path.exists(gesture_path):
                        images = len([f for f in os.listdir(gesture_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        total_images += images
            
            if total_images == 0:
                raise ValueError("No images found in subject folders! Check your dataset structure and image file extensions.")
            
            return self.create_unified_dataset(dataset_path)
            
        elif gesture_folders:
            print(f"✅ Found direct gesture structure with {len(gesture_folders)} classes")
            # Count images in direct structure
            for gesture_class in gesture_folders:
                gesture_path = os.path.join(dataset_path, gesture_class)
                images = len([f for f in os.listdir(gesture_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total_images += images
            
            if total_images == 0:
                raise ValueError("No images found in gesture folders! Check your image file extensions.")
            
            return dataset_path
        
    
    def create_unified_dataset(self, source_path):
        """Create unified dataset from subject-based structure"""
        unified_path = "dataset/unified_train"
        os.makedirs(unified_path, exist_ok=True)
        
        print("Creating unified dataset...")
        
        # Create gesture class directories
        for gesture_class in self.class_names:
            os.makedirs(os.path.join(unified_path, gesture_class), exist_ok=True)
        
        # Copy images from all subjects
        subject_folders = [d for d in os.listdir(source_path) 
                          if os.path.isdir(os.path.join(source_path, d)) and d.isdigit()]
        
        total_copied = 0
        for subject in subject_folders:
            for gesture_class in self.class_names:
                source_gesture_path = os.path.join(source_path, subject, gesture_class)
                target_gesture_path = os.path.join(unified_path, gesture_class)
                
                if os.path.exists(source_gesture_path):
                    images = [f for f in os.listdir(source_gesture_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    for image in images:
                        source_file = os.path.join(source_gesture_path, image)
                        # Add subject prefix to avoid filename conflicts
                        new_filename = f"{subject}_{image}"
                        target_file = os.path.join(target_gesture_path, new_filename)
                        
                        if not os.path.exists(target_file):
                            shutil.copy2(source_file, target_file)
                            total_copied += 1
        
        print(f"✅ Created unified dataset with {total_copied} images at: {unified_path}")
        return unified_path
    
    def prepare_data(self, dataset_path):
        """Prepare training and validation datasets"""
        print("Preparing dataset...")
        
        # Check and fix dataset structure
        processed_dataset_path = self.check_dataset_structure(dataset_path)
        
        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            processed_dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names
        )
        
        # Validation data generator
        validation_generator = train_datagen.flow_from_directory(
            processed_dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Classes: {list(train_generator.class_indices.keys())}")
        
        if train_generator.samples == 0:
            raise ValueError("No training samples found! Please check your dataset structure and image files.")
        
        return train_generator, validation_generator
    
    def create_model(self):
        """Create CNN model for hand gesture recognition"""
        print("Creating model...")
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling instead of flatten to reduce parameters
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_generator, validation_generator, epochs=50):
        """Train the model"""
        print("Starting training...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'models/best_gesture_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def evaluate_model(self, validation_generator):
        """Evaluate model and create confusion matrix"""
        print("Evaluating model...")
        
        # Get predictions
        validation_generator.reset()
        predictions = self.model.predict(validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = validation_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return cm
    
    def load_model(self, model_path):
        """Load trained model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def predict_gesture(self, image_path):
        """Predict gesture from image"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        # Make prediction
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        
        return self.class_names[predicted_class], confidence
    
    def predict_from_camera(self):
        """Real-time gesture recognition from camera"""
        cap = cv2.VideoCapture(0)
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (self.img_width, self.img_height))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)
            
            # Make prediction
            prediction = self.model.predict(input_frame)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
            
            # Display result
            gesture_name = self.class_names[predicted_class]
            text = f"{gesture_name}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main training pipeline"""
    # Initialize recognizer
    recognizer = HandGestureRecognizer()
    
    # Prepare data
    dataset_path = "dataset/train"
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found!")
        return
    
    train_gen, val_gen = recognizer.prepare_data(dataset_path)
    
    # Create and train model
    model = recognizer.create_model()
    print(model.summary())
    
    # Train model
    history = recognizer.train_model(train_gen, val_gen, epochs=50)
    
    # Plot results
    recognizer.plot_training_history(history)
    
    
    # Evaluate model
    recognizer.evaluate_model(val_gen)
    
    # Save class names for deployment
    with open('models/class_names.json', 'w') as f:
        json.dump(recognizer.class_names, f)
    
    print("Training completed! Model saved as 'models/best_gesture_model.h5'")

if __name__ == "__main__":
    main()