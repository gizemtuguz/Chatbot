import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Model ve tokenizer'ı yükle
model_path = "./trained_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Orijinal veri DataFrame'ini yükleme
df_original = pd.read_pickle('data_original.pkl')

def parse_table_data(table):
    try:
        headers = table.get('Headers', [])
        rows = table.get('Rows', [])
        table_df = pd.DataFrame(rows, columns=headers)
        return table_df
    except Exception as e:
        print(f"Error processing table data: {e}")
        return None

def table_to_html(table):
    headers = table.get('Headers', [])
    rows = table.get('Rows', [])
    table_df = pd.DataFrame(rows, columns=headers)
    table_html = table_df.to_html(classes='styled-table', index=False)
    return table_html

def get_chatbot_response(text, df_original):
    text = text.strip().lower()

    best_title_similarity = 0
    best_title = None

    vectorizer = TfidfVectorizer()

    # Başlıklar arasındaki benzerlikleri hesapla
    for _, row in df_original.iterrows():
        title = row['Title']
        corpus = [title, text]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        if similarity > best_title_similarity:
            best_title_similarity = similarity
            best_title = title

    if best_title_similarity < 0.2:  # Eşik değeri
        return "Bu konuda size daha iyi yardımcı olabilmem için daha detaylı bilgi veriniz lütfen"

    # En yüksek benzerlik skoru olan başlık altındaki tüm içerikleri göster
    most_relevant_group = df_original[df_original['Title'] == best_title]

    response = ""
    for _, row in most_relevant_group.iterrows():
        content_list = row['Content']
        for content_item in content_list:
            if 'Paragraph' in content_item:
                response += f"{content_item['Paragraph']}\n"
            if 'Image' in content_item:
                response += f'<img src="{content_item["Image"]}" alt="Image">\n'
            if 'Table' in content_item:
                table_content = content_item.get('Table', None)
                if table_content:
                    table_html = table_to_html(table_content)
                    response += f"{table_html}\n"
    return response