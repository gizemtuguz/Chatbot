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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Prepare data for training
def prepare_data(filepath, tokenizer):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    titles = []
    contents = []

    for entry in data:
        title = entry['Title']
        content_list = entry['Content']

        # Iterate through each content item in the content list
        for content_item in content_list:
            if 'Paragraph' in content_item:
                paragraph = content_item['Paragraph'].strip()
                if paragraph:  # Ensure the paragraph is not empty
                    titles.append(title)
                    contents.append(paragraph)
            if 'List' in content_item:
                for list_item in content_item['List']:
                    if isinstance(list_item, str):
                        list_content = list_item.strip()
                        if list_content:
                            titles.append(title)
                            contents.append(list_content)
                    elif isinstance(list_item, dict) and 'row' in list_item:
                        list_content = list_item['row'].strip()
                        if list_content:
                            titles.append(title)
                            contents.append(list_content)
            if 'Blockquote' in content_item:
                blockquote = content_item['Blockquote'].strip()
                if blockquote:  # Ensure the blockquote is not empty
                    titles.append(title)
                    contents.append(blockquote)

    # Combine Title and Content for training
    texts = [f"{title} {content}" for title, content in zip(titles, contents)]

    # Ensure consistency
    if len(texts) != len(contents):
        raise ValueError(f"Number of texts ({len(texts)}) does not match number of content items ({len(contents)}).")

    # Split the data into training and evaluation sets
    X_train, X_eval, y_train, y_eval = train_test_split(texts, contents, test_size=0.2, random_state=42)

    # Tokenize the training and evaluation sets separately
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    eval_encodings = tokenizer(X_eval, truncation=True, padding=True, max_length=512)

    # Since this is a plain text model, we'll use binary labels
    y_train_labels = [1] * len(y_train)  # Assign a label of 1 to all training examples
    y_eval_labels = [1] * len(y_eval)  # Assign a label of 1 to all evaluation examples

    # Create the datasets
    train_dataset = CustomDataset(train_encodings, y_train_labels)
    eval_dataset = CustomDataset(eval_encodings, y_eval_labels)

    return train_dataset, eval_dataset

# Train function
def train_model(train_dataset, eval_dataset, model_name, output_dir, num_labels=2, learning_rate=1e-5, epochs=3):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",  # Set evaluation strategy to "epoch"
        save_strategy="epoch",  # Align save strategy with evaluation strategy
        learning_rate=learning_rate,
        load_best_model_at_end=True,  # Load the best model at the end of training
    )

    model = model_name.from_pretrained(model_name.name_or_path, num_labels=num_labels)

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

if __name__ == "__main__":
    # Model and tokenizer for plain text data
    model_name_plain = RobertaForSequenceClassification.from_pretrained("roberta-base")
    tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Prepare data and train for plain text model
    train_dataset_plain, eval_dataset_plain = prepare_data('./data_plain_text.json', tokenizer_roberta)
    train_model(train_dataset_plain, eval_dataset_plain, model_name_plain, './trained_model_plain_text')

    print("Training completed for plain text model.")
