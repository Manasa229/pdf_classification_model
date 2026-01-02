import pandas as pd

from bert_processor import classify_with_bert

# Load your data
df = pd.read_csv('training/data_set/dataset.csv', names=['text', 'document_type'])

correct = 0
total = 0
confidences = []

for idx, row in df.iterrows():
    predicted_class, confidence = classify_with_bert(row['text'])
    actual_class = row['document_type']

    is_correct = (predicted_class == actual_class)
    correct += is_correct
    total += 1
    confidences.append(confidence)

    print(f"Text: {row['text'][:50]}...")
    print(
        f"Actual: {actual_class} | Predicted: {predicted_class} | Confidence: {confidence:.3f} | {'✓' if is_correct else '✗'}")
    print("-" * 80)

accuracy = correct / total
avg_confidence = sum(confidences) / len(confidences)

print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")
print(f"Average Adjusted Confidence: {avg_confidence:.3f}")