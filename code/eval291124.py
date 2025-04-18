import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load data from JSON file
with open("qwen_results.json", "r") as file:  # Replace "data.json" with your actual file path
    data = json.load(file)

# Preprocess the predicted and ground truth labels
predicted_labels = [item["predicted_answer"].replace("Classification: ", "").strip() for item in data]
ground_truth_labels = [item["ground_truth"].strip() for item in data]

# Calculate metrics
precision = precision_score(ground_truth_labels, predicted_labels, average='weighted')
recall = recall_score(ground_truth_labels, predicted_labels, average='weighted')
f1 = f1_score(ground_truth_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(ground_truth_labels, predicted_labels)

# Print metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
