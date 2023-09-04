import pickle
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the trained model
model = load_model('Cats_vs_Dogs.model')

# Load the test data and labels (you should have a separate test dataset)
X_test = pickle.load(open('X.pkl', 'rb'))
y_test = pickle.load(open('y.pkl', 'rb'))

# Normalize the test data
X_test = X_test / 255

# Reshape the test data if needed
X_test = X_test.reshape(-1, 60, 60, 1)

# Evaluate the model on the test data
y_pred = model.predict(X_test)

# Convert the one-hot encoded predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy}")

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Generate a classification report
class_report = classification_report(y_test, y_pred_classes)
print("Classification Report:")
print(class_report)
