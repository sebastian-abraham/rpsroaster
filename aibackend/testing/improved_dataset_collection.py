import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time

# Initialize MediaPipe Hands
# Use static_image_mode=False for video stream
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Normalization Function (Your excellent version) ---
def normalize_landmarks(landmarks):
    """
    Normalizes hand landmarks to be invariant to position and scale.
    :param landmarks: A list of (x, y, z) tuples for each landmark.
    :return: A flattened and normalized numpy array of landmarks.
    """
    # Use the wrist as the origin
    wrist = np.array(landmarks[0])
    shifted = np.array(landmarks) - wrist
    
    # Find the maximum distance from the wrist to any other landmark
    # np.linalg.norm computes the Euclidean distance
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    
    # Avoid division by zero if hand is not detected properly
    if max_dist == 0:
        return np.zeros(len(landmarks) * 3)
        
    # Normalize by dividing by the max distance
    normalized = shifted / max_dist
    
    # Flatten the array to a single vector for the model
    return normalized.flatten()

# --- Live Data Collection ---
cap = cv2.VideoCapture(0)

data = []
labels = []
sample_counts = {"rock": 0, "paper": 0, "scissors": 0}
classes = {"rock": 0, "paper": 1, "scissors": 2}

print("Starting data collection...")
print("Press 'r' for Rock, 'p' for Paper, 's' for Scissors. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hands
    results = hands.process(rgb_frame)

    # Display instructions and counts
    cv2.putText(frame, "Press 'r'(Rock), 'p'(Paper), 's'(Scissors)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to Quit & Save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_pos = 90
    for gesture, count in sample_counts.items():
        cv2.putText(frame, f"Collected {gesture}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks for visual feedback
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check for key presses to collect data
            key = cv2.waitKey(1) & 0xFF

            gesture_to_collect = None
            if key == ord('r'):
                gesture_to_collect = "rock"
            elif key == ord('p'):
                gesture_to_collect = "paper"
            elif key == ord('s'):
                gesture_to_collect = "scissors"
            elif key == ord('q'):
                break # Will be handled outside the inner loop

            if gesture_to_collect:
                # Extract coordinates and normalize them
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized_coords = normalize_landmarks(coords)
                
                data.append(normalized_coords)
                labels.append(classes[gesture_to_collect])
                sample_counts[gesture_to_collect] += 1
                print(f"Collected sample for {gesture_to_collect}. Total: {sample_counts[gesture_to_collect]}")

    # Check for quit key outside the landmark detection loop as well
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Interactive Data Collection', frame)

# --- Cleanup and Save ---
print("Quitting...")
cap.release()
cv2.destroyAllWindows()
hands.close()

# Save the collected data to a file
if data:
    with open("rps_dataset.pkl", "wb") as f:
        pickle.dump((data, labels), f)
    print(f"Successfully saved dataset with {len(data)} samples to rps_dataset.pkl")
else:
    print("No data was collected. Nothing to save.")
