import json
import pandas as pd
import re
import string
from transformers import BertTokenizer

# Temizleme fonksiyonu (eğitim için)
def clean_text_for_training(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Noktalama işaretlerini kaldır
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# JSON dosyasını yükleme fonksiyonu
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Metinleri küçük parçalara bölme fonksiyonu
def split_text(text, tokenizer, max_len=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_len - 2):  # -2 for special tokens [CLS] and [SEP]
        chunk = tokens[i:i + max_len - 2]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
    return chunks

# Öncelikli veri kontrolü
def is_priority_data(title):
    return not title.endswith('?')

# Veriyi DataFrame'e dönüştürme (eğitim için temizlenmiş ve bölünmüş verilerle)
def extract_data_for_training(data, tokenizer, max_len=512):
    records = []
    for entry in data:
        title = entry.get('Title', '')
        is_priority = is_priority_data(title)
        content_list = entry.get('Content', [])
        for content in content_list:
            if 'Paragraph' in content:
                paragraph = clean_text_for_training(content['Paragraph'])
                split_paragraphs = split_text(paragraph, tokenizer, max_len)
                for split_paragraph in split_paragraphs:
                    records.append({'Title': title, 'Content Type': 'Paragraph', 'Content': split_paragraph, 'Priority': is_priority})
            elif 'Table' in content:
                table = content['Table']
                headers = table.get('Headers', [])
                rows = table.get('Rows', [])
                table_text = f"Headers: {headers}, Rows: {rows}"
                split_tables = split_text(table_text, tokenizer, max_len)
                for split_table in split_tables:
                    records.append({'Title': title, 'Content Type': 'Table', 'Content': split_table, 'Priority': is_priority})
    return pd.DataFrame(records)

# JSON dosyalarının yolları
file_paths = [
    '/Users/gizemtuguz/Desktop/deneme/filtered_data_kitaplık.json',
    '/Users/gizemtuguz/Desktop/deneme/filtered_data_kitaplık_eng.json',
    '/Users/gizemtuguz/Desktop/deneme/filtered_data_epaticom.json',
    '/Users/gizemtuguz/Desktop/deneme/basic_data.json'
]

# Tüm JSON dosyalarından verileri birleştirme
combined_data_list = []
for file_path in file_paths:
    data = load_json_file(file_path)
    combined_data_list.extend(data)

# Tokenizer'ı yükleme
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Veriyi DataFrame'e dönüştürme (data_original.pkl için)
df_original = pd.DataFrame(combined_data_list)

# DataFrame'i orijinal haliyle kaydetme
df_original.to_pickle('data_original.pkl')

# Verileri DataFrame'e çıkarma (eğitim için)
df_training = extract_data_for_training(combined_data_list, tokenizer, max_len=512)

# Ağırlıklandırma stratejisi: Priority True olanların önceliğini artırma
df_training['Weight'] = df_training['Priority'].apply(lambda x: 1 if x else 2)

# DataFrame'i kaydetme (eğitim verisi)
df_training.to_pickle('data_training.pkl')

print(df_training.head())
print(df_original.head())
print(df_training)