import cv2
import mediapipe as mp
import numpy as np

def normalize_landmarks(landmarks):
    # landmarks: list of (x, y, z)
    wrist = np.array(landmarks[0])  # landmark 0 = wrist
    shifted = np.array(landmarks) - wrist  # translate wrist to origin
    
    max_dist = np.max(np.linalg.norm(shifted, axis=1))  # farthest point from wrist
    normalized = shifted / max_dist  # scale so largest distance = 1
    
    return normalized.flatten()  # shape: (63,)

# Init MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,   # Live video, not static images
    max_num_hands=1,           # Detect only one hand for speed
    model_complexity=0,        # 0 = fastest, 1 = more accurate
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect & convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw landmarks + print coords
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            normalized_coords = normalize_landmarks(coords)
            print(normalized_coords)  # 63 numbers, ready for MLP
            max_dist = np.max(np.linalg.norm(np.array(coords).reshape(-1, 3), axis=1))
            print(max_dist)
    # Show video
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
