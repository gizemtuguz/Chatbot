import os
import json
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
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

# Prepare data for training with labels based on the first word of the title
def prepare_data(filepath, tokenizer):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    labels = []
    first_word_to_label = {}
    current_label = 1

    for entry in data:
        title = entry['Title']
        first_word = title.split()[0]  # Get the first word of the title

        if first_word not in first_word_to_label:
            first_word_to_label[first_word] = current_label
            current_label += 1

        label = first_word_to_label[first_word]

        content_list = entry['Content']
        combined_text = title

        for content_item in content_list:
            if 'Paragraph' in content_item:
                combined_text += " " + content_item['Paragraph'].strip()
            if 'List' in content_item:
                for list_item in content_item['List']:
                    if isinstance(list_item, str):
                        combined_text += " " + list_item.strip()
            if 'Blockquote' in content_item:
                combined_text += " " + content_item['Blockquote'].strip()

        texts.append(combined_text.strip())
        labels.append(label)

    # Split the data into training and evaluation sets
    X_train_texts, X_eval_texts, y_train, y_eval = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Tokenize the training and evaluation sets separately
    train_encodings = tokenizer(X_train_texts, truncation=True, padding=True, max_length=512)
    eval_encodings = tokenizer(X_eval_texts, truncation=True, padding=True, max_length=512)

    # Create the datasets
    train_dataset = CustomDataset(train_encodings, y_train)
    eval_dataset = CustomDataset(eval_encodings, y_eval)

    return train_dataset, eval_dataset

# Train function
def train_model(train_dataset, eval_dataset, model_name, tokenizer, output_dir, num_labels, learning_rate=3e-5, epochs=10):
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, 'logs')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=10,  
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        load_best_model_at_end=True,
    )

    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Eval results for {output_dir}: {eval_result}")

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Directory to save everything
    save_directory = './trained_model_plain_text'
    
    # Prepare data and train
    train_dataset_plain, eval_dataset_plain = prepare_data('./data_plain_text.json', tokenizer_roberta)
    num_labels = len(set(train_dataset_plain.labels))  # Set the number of unique labels
    train_model(train_dataset_plain, eval_dataset_plain, "roberta-base", tokenizer_roberta, save_directory, num_labels)

    print("Training completed for plain text model.")
