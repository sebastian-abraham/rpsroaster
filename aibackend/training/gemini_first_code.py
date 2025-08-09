import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 1. Data Collection and Preprocessing ---
def collect_data():
    """
    Captures hand landmarks for rock, paper, and scissors gestures.
    Saves the collected data to a file.
    """
    data = []
    labels = []
    gestures = {"rock": 0, "paper": 1, "scissors": 2}
    cap = cv2.VideoCapture(0)

    for gesture, label in gestures.items():
        print(f"Collecting data for: {gesture}")
        # Give user time to prepare
        time.sleep(3)
        
        # Collect 100 samples per gesture
        for i in range(100):
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and find hands
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # --- Data Normalization ---
                    # Get coordinates of the wrist (landmark 0) to use as a reference
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    
                    # Extract and normalize landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x - wrist_x, lm.y - wrist_y])
                    
                    data.append(landmarks)
                    labels.append(label)

            # Display the frame
            cv2.putText(frame, f"Collecting {gesture}: Sample {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data
    with open('hand_gesture_data.pkl', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data collection complete and saved.")

# --- 2. Model Training ---
def train_model():
    """
    Loads the collected data, trains an MLP classifier, and saves the model.
    """
    if not os.path.exists('hand_gesture_data.pkl'):
        print("Data file not found. Please run collect_data() first.")
        return

    with open('hand_gesture_data.pkl', 'rb') as f:
        dataset = pickle.load(f)

    X = np.array(dataset['data'])
    y = np.array(dataset['labels'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and train the MLP Classifier
    # Using a simple MLP for lightweight performance
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    with open('rps_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as rps_model.pkl")

# --- 3. Real-time Prediction ---
def play_game():
    """
    Uses the trained model to predict hand gestures in real-time and play
    Rock-Paper-Scissors against the user, always winning.
    """
    if not os.path.exists('rps_model.pkl'):
        print("Model not found. Please run train_model() first.")
        return

    with open('rps_model.pkl', 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    gesture_map = {0: "Rock", 1: "Paper", 2: "Scissors"}
    winning_moves = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
    
    last_prediction_time = time.time()
    prediction_interval = 1 # seconds
    user_move = ""
    bot_move = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Predict every 'prediction_interval' seconds
                if time.time() - last_prediction_time > prediction_interval:
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x - wrist_x, lm.y - wrist_y])

                    # Make a prediction
                    prediction = model.predict([landmarks])
                    user_move = gesture_map[prediction[0]]
                    bot_move = winning_moves[user_move]
                    
                    last_prediction_time = time.time()

        # Display the moves
        cv2.putText(frame, f"You: {user_move}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Bot: {bot_move}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if user_move and bot_move:
             cv2.putText(frame, "Bot wins!", (150, 250), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 3)


        cv2.imshow('Rock Paper Scissors', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":
    print("Choose an option:")
    print("1: Collect Data")
    print("2: Train Model")
    print("3: Play Game")
    
    choice = input("Enter choice (1/2/3): ")

    if choice == '1':
        collect_data()
    elif choice == '2':
        train_model()
    elif choice == '3':
        play_game()
    else:
        print("Invalid choice.")
