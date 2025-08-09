import pickle
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Dataset ---
# Ensure the 'rps_dataset.pkl' file from the collection script is in the same directory
try:
    with open("rps_dataset.pkl", "rb") as f:
        # The data was saved as a tuple (data, labels)
        data, labels = pickle.load(f)
except FileNotFoundError:
    print("Error: rps_dataset.pkl not found.")
    print("Please run the interactive_data_collection.py script first to create the dataset.")
    exit()

print(f"Dataset loaded with {len(data)} samples.")

# --- 2. Split Data for Training and Testing ---
# stratify=y ensures that the proportion of each class is the same in train and test sets
# This is important for imbalanced datasets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- 3. Train the MLP Classifier ---
# Your MLP configuration is a good starting point.
# hidden_layer_sizes=(64, 32) means two hidden layers with 64 and 32 neurons.
# max_iter=500 is a reasonable number of iterations to allow for convergence.
print("Training the MLP model...")
clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
clf.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# --- Detailed Classification Report ---
# This shows precision, recall, and f1-score for each class (rock, paper, scissors)
print("\nClassification Report:")
class_names = ['Rock', 'Paper', 'Scissors']
print(classification_report(y_test, y_pred, target_names=class_names))

# --- Confusion Matrix ---
# This helps visualize where the model is getting confused.
# For example, how many times was "Rock" predicted as "Paper"?
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# --- 5. Save the Trained Model ---
# Using joblib is a good choice for scikit-learn models.
joblib.dump(clf, "rps_mlp_model_improved.pkl")
print("\nModel successfully saved to rps_mlp_model.pkl")
