import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
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

    texts = []
    labels = []

    for entry in data:
        content_list = entry['Content']
        combined_text = ""

        # Combine all content into a single string
        for content_item in content_list:
            if 'Paragraph' in content_item:
                combined_text += " " + content_item['Paragraph']
            if 'List' in content_item:
                for list_item in content_item['List']:
                    if isinstance(list_item, str):
                        combined_text += " " + list_item
                    elif isinstance(list_item, dict) and 'row' in list_item:
                        combined_text += " " + list_item['row']
            if 'Table' in content_item:
                table_data = content_item['Table']
                if isinstance(table_data, dict) and 'Rows' in table_data:
                    for table_row in table_data['Rows']:
                        combined_text += " " + " | ".join(table_row)
                elif isinstance(table_data, list):
                    # Handle case where 'Table' might be a list of table entries
                    for table_entry in table_data:
                        if isinstance(table_entry, dict) and 'Rows' in table_entry:
                            for table_row in table_entry['Rows']:
                                combined_text += " " + " | ".join(table_row)
            if 'Blockquote' in content_item:
                combined_text += " " + content_item['Blockquote']

        combined_text = combined_text.strip()
        if combined_text:  # Ensure that we only add non-empty texts
            texts.append(combined_text)
            labels.append(1 if "?" in entry['Title'] else 0)  # Adjust based on the type of training (1 for questions, 0 for plain text)

    # Ensure that the number of texts matches the number of labels
    if len(texts) != len(labels):
        raise ValueError(f"Number of texts ({len(texts)}) does not match number of labels ({len(labels)}).")

    # Split the data into training and evaluation sets
    X_train, X_eval, y_train, y_eval = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Tokenize the training and evaluation sets separately
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    eval_encodings = tokenizer(X_eval, truncation=True, padding=True, max_length=512)

    # Create the datasets
    train_dataset = CustomDataset(train_encodings, y_train)
    eval_dataset = CustomDataset(eval_encodings, y_eval)

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
    # Model and tokenizer for question-answer data
    model_name_qa = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Prepare data and train for question-answer model
    print("Training question-answer model...")
    train_dataset_qa, eval_dataset_qa = prepare_data('./data_question_answer.json', tokenizer_bert)
    train_model(train_dataset_qa, eval_dataset_qa, model_name_qa, './trained_model_question_answer')
    
    # Once the question-answer model training is done, move to the next
    print("Training plain text model...")
    
    # Model and tokenizer for plain text data
    model_name_plain = RobertaForSequenceClassification.from_pretrained("roberta-base")
    tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Prepare data and train for plain text model
    train_dataset_plain, eval_dataset_plain = prepare_data('./data_plain_text.json', tokenizer_roberta)
    train_model(train_dataset_plain, eval_dataset_plain, model_name_plain, './trained_model_plain_text')

    print("Training completed for both models.")
