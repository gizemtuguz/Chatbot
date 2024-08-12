import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Eğitim ve test veri setlerini yükleme
df_training = pd.read_pickle('data_training.pkl')

# Eğitim ve test veri setlerini ayırma
X_train, X_test = train_test_split(df_training, test_size=0.2, random_state=42)

# Tokenizer ve model yükleme
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Verileri tokenleştirme
train_encodings = tokenizer(list(X_train['Content']), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test['Content']), truncation=True, padding=True, max_length=512)

# Dataset sınıfı
class CustomDataset(Dataset):
    def __init__(self, encodings, labels, weights):
        self.encodings = encodings
        self.labels = labels
        self.weights = weights

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['weight'] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Etiketleri ve ağırlıkları ayarlama
train_labels = [1 if p else 0 for p in X_train['Priority']]
test_labels = [1 if p else 0 for p in X_test['Priority']]
train_weights = list(X_train['Weight'])
test_weights = list(X_test['Weight'])

# Dataset oluşturma
train_dataset = CustomDataset(train_encodings, train_labels, train_weights)
test_dataset = CustomDataset(test_encodings, test_labels, test_weights)

# Eğitim argümanlarını ayarlama
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=2e-5,  # Öğrenme oranı
    lr_scheduler_type="linear",  # Öğrenme oranını adımlarla düşürme stratejisi
)

# Hesaplama fonksiyonunu ekleme
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

# Custom Trainer sınıfı (Focal Loss için)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        weights = inputs.get("weight")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Focal Loss hesaplama
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2 * ce_loss).mean()  # Focal Loss uygulandı
        
        return (focal_loss, outputs) if return_outputs else focal_loss

# Trainer ayarlama
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Eğitimi başlatma
trainer.train()

# Modeli kaydetme
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Modeli değerlendirme
metrics = trainer.evaluate()
print(metrics)
