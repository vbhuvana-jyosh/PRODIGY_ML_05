# PRODIGY_ML_05
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
from sklearn.preprocessing import LabelEncoder
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define gesture labels (customize as needed)
GESTURE_LABELS = ['fist', 'open_palm', 'point_up', 'thumbs_up', 'peace_sign']

class GestureRecognizer:
    def __init__(self, model_path=None):
        self.model = self._build_model()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(GESTURE_LABELS)
        
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)

    def _build_model(self):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(21*3,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(lenInstances: 5
            Dense(len(GESTURE_LABELS), activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def process_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Normalize landmarks
                landmarks = np.array(landmarks)
                landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
                
                # Predict gesture
                prediction = self.model.predict(landmarks.reshape(1, -1))
                gesture = self.label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
                
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Display gesture
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
        return frame

    def collect_training_data(self, output_dir, samples_per_gesture=100):
        cap = cv2.VideoCapture(0)
        data = []
        labels = []
        
        for gesture in GESTURE_LABELS:
            print(f"Collecting data for {gesture}. Press 'c' to start, 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = self.process_frame(frame)
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    print(f"Collecting {samples_per_gesture} samples...")
                    for i in range(samples_per_gesture):
                        ret, frame = cap.read()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(frame_rgb)
                        
                        if results.multi_hand_landmarks:
                            landmarks = []
                            for lm in results.multi_hand_landmarks[0].landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])
                            data.append(landmarks)
                            labels.append(gesture)
                        
                        cv2.imshow('Data Collection', frame)
                        cv2.waitKey(10)
                    
                    print(f"Finished collecting {gesture}")
                    break
                elif key == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        np.save(os.path.join(output_dir, 'gesture_data.npy'), np.array(data))
        np.save(os.path.join(output_dir, 'gesture_labels.npy'), np.array(labels))
        
        return np.array(data), np.array(labels)

    def train_model(self, data, labels, epochs=50, validation_split=0.2):
        # Encode labels
        encoded_labels = self.label_encoder.transform(labels)
        encoded_labels = tf.keras.utils.to_categorical(encoded_labels)
        
        # Train model
        history = self.model.fit(
            data, encoded_labels,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32
        )
        
        return history

    def save_model(self, output_dir):
        self.model.save(os.path.join(output_dir, 'gesture_model.h5'))
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

def main():
    recognizer = GestureRecognizer()
    
    # Collect training data
    data, labels = recognizer.collect_training_data('gesture_data')
    
    # Train model
    history = recognizer.train_model(data, labels)
    
    # Save model
    recognizer.save_model('gesture_data')
    
    # Real-time testing
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = recognizer.process_frame(frame)
        cv2.imshow('Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

    
