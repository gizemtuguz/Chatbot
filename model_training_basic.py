import os
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Custom Dataset class
class BasicDataDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Labeling strategy based on content
def label_based_on_content(title):
    if any(word in title.lower() for word in ["hello", "hi", "merhaba", "günaydın", "selam"]):
        return 1  # Label 1 for greetings
    elif any(word in title.lower() for word in ["goodbye", "güle güle", "hoşça kal", "see you", "bye"]):
        return 2  # Label 2 for goodbyes
    elif any(word in title.lower() for word in ["thank you", "thanks", "teşekkür"]):
        return 3  # Label 3 for thanks
    else:
        return 4  # Label 4 for other types of content

# Load the basic_data.json and label the data
def load_and_label_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    labels = []

    for entry in data:
        title = entry['Title']
        label = label_based_on_content(title)
        content = " ".join([content_item['Paragraph'] for content_item in entry['Content'] if 'Paragraph' in content_item])
        texts.append(content)
        labels.append(label)

    return texts, labels

# Main function to load, process, and train the model
def main():
    # Create the directory to save everything
    base_output_dir = './basic_data_trained_model'
    os.makedirs(base_output_dir, exist_ok=True)

    # Load and label the data
    filepath = './basic_data.json'
    texts, labels = load_and_label_data(filepath)

    # Save the labeled data to the output directory
    labeled_data_path = os.path.join(base_output_dir, 'labeled_basic_data.json')
    with open(labeled_data_path, 'w', encoding='utf-8') as f:
        json.dump({"texts": texts, "labels": labels}, f, ensure_ascii=False, indent=4)
    print(f"Labeled data saved to {labeled_data_path}")

    # Split the data into training and evaluation sets
    X_train_texts, X_eval_texts, y_train, y_eval = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Create the datasets
    train_dataset = BasicDataDataset(X_train_texts, y_train, tokenizer)
    eval_dataset = BasicDataDataset(X_eval_texts, y_eval, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(base_output_dir, 'results'),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(base_output_dir, 'logs'),  # Save logs to this directory
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize the model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Eval results: {eval_result}")

    # Save the model and tokenizer
    model_save_path = os.path.join(base_output_dir, 'trained_distilbert_model')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

if __name__ == "__main__":
    main()
