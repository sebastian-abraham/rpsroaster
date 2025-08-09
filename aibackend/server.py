import asyncio
import websockets
import cv2
import numpy as np
import joblib
import base64
import json

# --- 1. Load Model and Initialize ---
try:
    clf = joblib.load("rps_mlp_model_improved.pkl")
except FileNotFoundError:
    print("Error: rps_mlp_model.pkl not found. Please train the model first.")
    exit()

# MediaPipe setup
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Label and game logic
classes = {0: "Rock", 1: "Paper", 2: "Scissors"}
winning_moves = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}

# --- 2. Normalization Function ---
def normalize_landmarks(landmarks):
    wrist = np.array(landmarks[0])
    shifted = np.array(landmarks) - wrist
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    if max_dist == 0:
        return np.zeros(len(landmarks) * 3)
    normalized = shifted / max_dist
    return normalized.flatten()

# --- 3. WebSocket Handler ---
async def handle_prediction(websocket):
    """
    Receives image data from a client, processes it, and sends back a prediction.
    """
    print("Client connected.")
    try:
        async for message in websocket:
            # The message is a base64 encoded image string
            # Decode the base64 string
            img_data = base64.b64decode(message.split(',')[1])
            
            # Convert to a numpy array
            np_arr = np.frombuffer(img_data, np.uint8)
            
            # Decode the numpy array into an image
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # --- Perform Prediction (same logic as before) ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            response = {
                "user_move": "",
                "bot_move": "",
                "confidence": 0.0
            }

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    normalized_coords = normalize_landmarks(coords)
                    
                    pred_index = clf.predict([normalized_coords])[0]
                    confidence = max(clf.predict_proba([normalized_coords])[0])
                    
                    if confidence > 0.85:
                        user_move = classes[pred_index]
                        bot_move = winning_moves.get(user_move, "")
                        response = {
                            "user_move": user_move,
                            "bot_move": bot_move,
                            "confidence": float(confidence) # Ensure it's JSON serializable
                        }

            # Send the JSON response back to the client
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- 4. Start the Server ---
async def main():
    # Change "localhost" to "0.0.0.0" to allow connections from other devices on your network
    async with websockets.serve(handle_prediction, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())

