from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "You are invited to the Annual Tech Conference on Sept 5th, 2025 at 10 AM."

labels = ["Event", "Not Event"]

result = classifier(text, candidate_labels=labels)
print(result)
