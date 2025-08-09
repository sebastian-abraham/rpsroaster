import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained MLP model
clf = joblib.load("rps_mlp_model.pkl")

# Label mapping
classes = {0: "Rock", 1: "Paper", 2: "Scissors"}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Normalization function
def normalize_landmarks(landmarks):
    wrist = np.array(landmarks[0])
    shifted = np.array(landmarks) - wrist
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    normalized = shifted / max_dist
    return normalized.flatten()

# Webcam loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            normalized_coords = normalize_landmarks(coords)

            pred = clf.predict([normalized_coords])[0]
            conf = max(clf.predict_proba([normalized_coords])[0])

            if conf > 0.8:  # only show if confident
                cv2.putText(frame, f"{classes[pred]} ({conf:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("RPS Prediction", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
