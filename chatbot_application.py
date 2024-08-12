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

# Tabloları başlık ve içerik olarak DataFrame'e çevirir
def parse_table_data(table):
    try:
        headers = table.get('Headers', [])
        rows = table.get('Rows', [])
        table_df = pd.DataFrame(rows, columns=headers)
        return table_df
    except Exception as e:
        print(f"Error processing table data: {e}")
        return None

# DataFrame'e çevrilen tablo bilgileri HTML konseptine çevirilir
def table_to_html(table):
    headers = table.get('Headers', [])
    rows = table.get('Rows', [])
    table_df = pd.DataFrame(rows, columns=headers)
    table_html = table_df.to_html(classes='styled-table', index=False)
    return table_html

def get_chatbot_response(text, df_original):
    text = text.strip().lower()

    best_similarity = 0
    best_title = None
    best_content = None

    vectorizer = TfidfVectorizer()

    #başlıklar ve içerikler arasındaki benzerlikleri hesapla
    for _, row in df_original.iterrows():
        title = row['Title']
        content_list = row['Content']
        title_similarity = cosine_similarity(vectorizer.fit_transform([title, text]))[0, 1]

        #her başlık altındaki içeriklerle de benzerlik hesapla
        for content_item in content_list:
            content_text = ""
            if 'Paragraph' in content_item:
                content_text = content_item['Paragraph']
            elif 'Table' in content_item:
                table_content = content_item.get('Table', None)
                if table_content:
                    table_text = f"Headers: {table_content.get('Headers', [])}, Rows: {table_content.get('Rows', [])}"
                    content_text = table_text
            elif 'Image' in content_item:
                content_text = f"Image: {content_item['Image']}"

            if content_text:
                content_similarity = cosine_similarity(vectorizer.fit_transform([content_text, text]))[0, 1]
                combined_similarity = (title_similarity + content_similarity) / 2

                if combined_similarity > best_similarity:
                    best_similarity = combined_similarity
                    best_title = title
                    best_content = content_item

    if best_similarity < 0.2:  # Eşik değeri
        return "Bu konuda size daha iyi yardımcı olabilmem için daha detaylı bilgi veriniz lütfen"

    #en yüksek benzerlik skoru olan başlık ve içerik için yanıt oluştur
    response = ""
    if best_content:
        if 'Paragraph' in best_content:
            response += f"{best_content['Paragraph']}\n"
        if 'Image' in best_content:
            response += f'<img src="{best_content["Image"]}" alt="Image">\n'
        if 'Table' in best_content:
            table_content = best_content.get('Table', None)
            if table_content:
                table_html = table_to_html(table_content)
                response += f"{table_html}\n"

    return response
