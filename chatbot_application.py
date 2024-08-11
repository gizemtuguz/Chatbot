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

# Konuşma bağlamı için bir dictionary
conversation_context = {}

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

def update_conversation_context(user_id, new_message, conversation_context):
    if user_id not in conversation_context:
        conversation_context[user_id] = []
    conversation_context[user_id].append(new_message)

def generate_response_with_context(context, df_original):
    # Karmaşıklık kontrolü (örnek olarak metin uzunluğuna bakılabilir)
    if len(context.split()) > 100:  
        return "Bu konuda size daha iyi yardımcı olabilmem için daha basit bir soru sorabilirsiniz."
    
    # Burada mevcut context'e dayalı yanıt üretme
    best_title_similarity = 0
    best_title = None

    vectorizer = TfidfVectorizer()

    # Başlıklar arasındaki benzerlikleri hesapla
    for _, row in df_original.iterrows():
        title = row['Title']
        corpus = [title, context]
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
                response += f'<img src="{content_item["Image"]}" alt="Image" style="max-width: 100%; height: auto;">\n'
            if 'Table' in content_item:
                table_content = content_item.get('Table', None)
                if table_content:
                    table_html = table_to_html(table_content)
                    response += f"{table_html}\n"
    return response

def get_chatbot_response(user_id, text, df_original, conversation_context):
    update_conversation_context(user_id, text, conversation_context)
    
    # Mevcut konuşma bağlamını alma
    context = " ".join(conversation_context[user_id])
    
    # Bağlama dayalı yanıt üretme
    response = generate_response_with_context(context, df_original)
    
    # Yanıtı bağlama ekleme
    update_conversation_context(user_id, response, conversation_context)
    
    return response
