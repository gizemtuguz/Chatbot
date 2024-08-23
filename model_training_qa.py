import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Custom Dataset class
class CustomQADataset(Dataset):
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

# Labeling strategy based on your criteria
def label_based_on_title(title):
    if "nedir" in title.lower() or "what is the meaning" in title.lower():
        return 1  # Label 1
    elif "nasıl" in title.lower() or "how" in title.lower():
        return 2  # Label 2
    elif "bilir miyim" in title.lower() or "can i" in title.lower():
        return 3  # Label 3
    elif "nere" in title.lower() or "where" in title.lower():
        return 4  # Label 4
    else:
        return 5  # Label 5 for all other cases

# Prepare data for training
def prepare_qa_data(filepath, tokenizer):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    labels = []

    for entry in data:
        title = entry['Title']
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
        labels.append(label_based_on_title(title))

    # Split the data into training and evaluation sets
    X_train_texts, X_eval_texts, y_train, y_eval = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Tokenize the training and evaluation sets separately
    train_encodings = tokenizer(X_train_texts, truncation=True, padding=True, max_length=512)
    eval_encodings = tokenizer(X_eval_texts, truncation=True, padding=True, max_length=512)

    # Create the datasets
    train_dataset = CustomQADataset(train_encodings, y_train)
    eval_dataset = CustomQADataset(eval_encodings, y_eval)

    return train_dataset, eval_dataset, len(set(labels))

# Train function
def train_qa_model(train_dataset, eval_dataset, model_name, tokenizer, output_dir, num_labels, learning_rate=3e-5, epochs=5):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        load_best_model_at_end=True,
    )

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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

    # Model ve tokenizer'ı kaydetme
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Prepare data and train
    train_dataset_qa, eval_dataset_qa, num_labels_qa = prepare_qa_data('./data_question_answer.json', tokenizer_bert)
    train_qa_model(train_dataset_qa, eval_dataset_qa, "bert-base-uncased", tokenizer_bert, './trained_model_qa', num_labels_qa)

    print("Training completed for QA model with BERT.")
