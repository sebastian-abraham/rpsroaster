import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
data = []
labels = []

def normalize_landmarks(landmarks):
    wrist = np.array(landmarks[0])
    shifted = np.array(landmarks) - wrist
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    normalized = shifted / max_dist
    return normalized.flatten()

base_dir = "C:/Users/LENOVO/Desktop/projects/rpsroaster/aibackend/testing/dataset"
classes = ["rock", "paper", "scissors"]

for label, cls in enumerate(classes):
    folder = os.path.join(base_dir, cls)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized_coords = normalize_landmarks(coords)
                data.append(normalized_coords)
                labels.append(label)

hands.close()

with open("rps_dataset.pkl", "wb") as f:
    pickle.dump((data, labels), f)

print(f"Saved dataset with {len(data)} samples")
