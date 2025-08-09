import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# --- 1. Load the Trained Model and Setup ---
try:
    # Load the model you trained with the enhanced script
    clf = joblib.load("rps_mlp_model_improved.pkl")
except FileNotFoundError:
    print("Error: rps_mlp_model.pkl not found.")
    print("Please run the training script first to create the model file.")
    exit()

# Label and game logic mappings
classes = {0: "Rock", 1: "Paper", 2: "Scissors"}
winning_moves = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}

# --- 2. MediaPipe Hand Tracking Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,      # Use False for video stream
    max_num_hands=1,              # We only need to track one hand
    min_detection_confidence=0.7, # Higher confidence for more stable detection
    min_tracking_confidence=0.5
)

# --- 3. Normalization Function (Your superior version) ---
def normalize_landmarks(landmarks):
    """
    Normalizes hand landmarks to be invariant to position and scale.
    """
    wrist = np.array(landmarks[0])
    shifted = np.array(landmarks) - wrist
    
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    if max_dist == 0:
        return np.zeros(len(landmarks) * 3) # Return zero vector if no hand detected
        
    normalized = shifted / max_dist
    return normalized.flatten()

# --- 4. Main Game Loop ---
cap = cv2.VideoCapture(0)

# Variables to hold the current moves
user_move = ""
bot_move = ""
last_user_move = ""
display_winner_until = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a selfie-view
    frame = cv2.flip(frame, 1)
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
    results = hands.process(rgb_frame)

    # Reset moves if no hand is detected
    if not results.multi_hand_landmarks:
        user_move = ""
        bot_move = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- Prediction Logic ---
            # Extract and normalize coordinates
            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            normalized_coords = normalize_landmarks(coords)

            # Get prediction and confidence from the model
            pred_index = clf.predict([normalized_coords])[0]
            confidence = max(clf.predict_proba([normalized_coords])[0])

            # Only update moves if confidence is high enough
            if confidence > 0.85: # Using a slightly higher threshold for stability
                predicted_move = classes[pred_index]
                # Check if the move has changed to trigger the "winner" text
                if predicted_move != last_user_move:
                    user_move = predicted_move
                    bot_move = winning_moves[user_move]
                    last_user_move = user_move
                    # Set a timer to display the "Bot Wins!" message for 2 seconds
                    display_winner_until = time.time() + 2

    # --- Display Text on Screen ---
    # Display the user's detected move
    cv2.putText(frame, f"You Showed: {user_move}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Display the bot's winning move
    cv2.putText(frame, f"Bot Plays: {bot_move}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the "Bot Wins!" message for a limited time
    if time.time() < display_winner_until:
        cv2.putText(frame, "Bot Wins!", (frame.shape[1] // 4, frame.shape[0] // 2), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)

    # Show the final frame
    cv2.imshow("Rock Paper Scissors - Bot Always Wins!", frame)

    # Press ESC (ASCII 27) to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- 5. Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
