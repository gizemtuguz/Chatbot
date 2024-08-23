from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import langdetect

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG)

# JSON dosyalarını yükleme
with open('basic_data.json', 'r', encoding='utf-8') as f:
    basic_data = json.load(f)

with open('filtered_data_epaticom.json', 'r', encoding='utf-8') as f:
    epaticom_data = json.load(f)

with open('filtered_data_kitaplık.json', 'r', encoding='utf-8') as f:
    kitaplık_data_tr = json.load(f)

with open('filtered_data_kitaplık_eng.json', 'r', encoding='utf-8') as f:
    kitaplık_data_eng = json.load(f)

# Flask uygulaması oluşturma
app = Flask(__name__)
CORS(app)  # CORS desteğini etkinleştir

# Model ve tokenizer yükleme fonksiyonu
def load_model_and_tokenizer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Modelleri ve tokenizerları yükleme
model_1, tokenizer_1 = load_model_and_tokenizer('trained_model_qa')
model_2, tokenizer_2 = load_model_and_tokenizer('trained_model_plain_text')
model_3, tokenizer_3 = load_model_and_tokenizer('basic_data_trained_model/trained_distilbert_model')

# Model tahmin fonksiyonu
def predict_with_model(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    confidence_score = torch.softmax(outputs.logits, dim=-1).max().item()
    logging.debug(f"Confidence Score: {confidence_score}")  # Güven skorunu logla
    return confidence_score

# Girdi cümlesinin dilini tespit etme fonksiyonu
def detect_language(input_text):
    detected_language = langdetect.detect(input_text)
    logging.debug(f"Detected Language: {detected_language} for input: {input_text}")
    return detected_language

# Tf-idf ile benzerlik hesaplama fonksiyonu
def calculate_similarity(input_text, text_list):
    vectorizer = TfidfVectorizer().fit_transform([input_text] + text_list)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    return cosine_matrix[0, 1:]  # İlk vektör (kullanıcı girişi) ile diğerleri arasındaki benzerlikler

# JSON verilerini HTML formatına dönüştürme fonksiyonu
def format_json_to_html(content_elements):
    html_content = ""
    
    for element in content_elements:
        if isinstance(element, dict):
            if 'Paragraph' in element:
                html_content += f"<p>{element['Paragraph']}</p>"
            elif 'Table' in element:
                table = element['Table']
                if isinstance(table, dict):  # Check if table is a dictionary
                    headers = "".join([f"<th>{header}</th>" for header in table.get('Headers', [])])
                    rows = "".join([f"<tr>{''.join([f'<td>{cell}</td>' for cell in row])}</tr>" for row in table.get('Rows', [])])
                    html_content += f"<table border='1'><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"
                elif isinstance(table, list):  # Handle if table is a list
                    rows = "".join([f"<tr>{''.join([f'<td>{cell}</td>' for cell in row])}</tr>" for row in table])
                    html_content += f"<table border='1'><tbody>{rows}</tbody></table>"
            elif 'List' in element:
                list_items = "".join([f"<li>{item}</li>" for item in element['List']])
                html_content += f"<ul>{list_items}</ul>"
            elif 'Blockquote' in element:
                html_content += f"<blockquote>{element['Blockquote']}</blockquote>"
            elif 'Image' in element:
                image = element['Image']
                if isinstance(image, dict):  # Ensure image is a dictionary
                    src = image.get('src', '')
                    alt = image.get('alt', '')
                    html_content += f"<img src='{src}' alt='{alt}' />"
                elif isinstance(image, str):  # Handle case where image is a string
                    html_content += f"<img src='{image}' alt='' />"
        else:
            logging.warning(f"Unrecognized element type: {element}")

    return html_content

# En iyi eşleşmeyi bulma fonksiyonu
def get_best_match_from_json(input_text, json_data, threshold=0.6):
    best_match = None
    highest_similarity = 0

    # JSON başlıklarını bir listeye toplayın
    titles = [item['Title'].lower() for item in json_data]

    # Tf-idf ile benzerlik hesaplayın
    similarities = calculate_similarity(input_text.lower(), titles)

    for idx, similarity in enumerate(similarities):
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = json_data[idx]['Content']

    logging.debug(f"Highest Similarity: {highest_similarity}, Best Match: {best_match}")

    # Eşik değeri altında kalan sonuçlar için hiçbir şey döndürme
    if highest_similarity < threshold:
        return "<p>Üzgünüm, bu soruya dair bir cevabım yok.</p>", highest_similarity

    # HTML'ye dönüştür ve döndür
    return format_json_to_html(best_match) if best_match else "<p>Üzgünüm, bu soruya dair bir cevabım yok.</p>", highest_similarity

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message').strip().lower()  # Normalize input
    
    # Dil tespiti ve veri seti seçimi
    detected_language = detect_language(user_input)
    logging.debug(f"Detected language: {detected_language}")

    if detected_language == 'en':
        response_kitaplik, similarity_kitaplik = get_best_match_from_json(user_input, kitaplık_data_eng)
    else:  # Varsayılan olarak Türkçe içerik
        response_kitaplik, similarity_kitaplik = get_best_match_from_json(user_input, kitaplık_data_tr)

    # Other model predictions and dataset searches
    confidence_1 = predict_with_model(model_1, tokenizer_1, user_input)
    confidence_2 = predict_with_model(model_2, tokenizer_2, user_input)
    confidence_3 = predict_with_model(model_3, tokenizer_3, user_input)

    logging.debug(f"Confidence Scores -> Model 1: {confidence_1}, Model 2: {confidence_2}, Model 3: {confidence_3}")
    
    response_basic, similarity_basic = get_best_match_from_json(user_input, basic_data)
    response_epaticom, similarity_epaticom = get_best_match_from_json(user_input, epaticom_data)

    # Determine the best response based on similarity
    if similarity_basic >= similarity_epaticom and similarity_basic >= similarity_kitaplik:
        best_response = response_basic
    elif similarity_epaticom >= similarity_basic and similarity_epaticom >= similarity_kitaplik:
        best_response = response_epaticom
    else:
        best_response = response_kitaplik

    logging.debug(f"Selected Response: {best_response}")  # Log the selected response

    return jsonify({'response': best_response})

# Flask sunucusunu çalıştırma
if __name__ == '__main__':
    app.run(debug=True, port=5001)
