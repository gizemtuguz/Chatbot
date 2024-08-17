import json
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your trained model and tokenizer
model_path = "./trained_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the original dataset from JSON instead of PKL
with open('combined_output.json', 'r', encoding='utf-8') as f:
    df_original = pd.DataFrame(json.load(f))

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
    best_similarity = 0
    best_row = None

    # Create a corpus from all titles and content, plus the input text
    corpus = [text] + df_original['Title'].tolist()
    for _, row in df_original.iterrows():
        for content_item in row['Content']:
            if 'Paragraph' in content_item:
                corpus.append(content_item['Paragraph'])
            elif 'Table' in content_item:
                table_content = content_item.get('Table', None)
                if table_content:
                    table_text = f"Headers: {table_content.get('Headers', [])}, Rows: {table_content.get('Rows', [])}"
                    corpus.append(table_text)
            elif 'Image' in content_item:
                corpus.append(f"Image: {content_item['Image']}")

    # Fit TF-IDF on the entire corpus and transform the input text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    text_vector = tfidf_matrix[0]  # First vector corresponds to the input text

    for idx, row in df_original.iterrows():
        title_vector = tfidf_matrix[idx + 1]  # Offset by 1 because the first entry is the input text
        title_similarity = cosine_similarity(title_vector, text_vector)[0, 0]

        for content_item in row['Content']:
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
                content_index = corpus.index(content_text)
                content_vector = tfidf_matrix[content_index]
                content_similarity = cosine_similarity(content_vector, text_vector)[0, 0]

                max_similarity = max(title_similarity, content_similarity)  # Use maximum similarity
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_row = row

    if best_similarity < 0.2:  # Threshold value
        return "Bu konuda size daha iyi yardımcı olabilmem için daha detaylı bilgi veriniz lütfen"

    response = ""
    if best_row is not None:
        content_list = best_row['Content']
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
