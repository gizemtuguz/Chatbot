from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_application import get_chatbot_response, df_original
import requests
import base64

app = Flask(__name__)
CORS(app)

# Resim URL'sini indirip base64 formatında döndüren yardımcı fonksiyon
def fetch_image_as_base64(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Hata oluşursa yakala
        image_bytes = response.content
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    except requests.RequestException as e:
        print(f"Error fetching image: {e}")
        return None

# Chatbot yanıtı
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json.get('message')
    print(f"Received message: {user_input}")
    
    try:
        response = get_chatbot_response(user_input, df_original)
        
        # Eğer yanıt bir img src içeriyorsa, resmi indir ve base64 olarak döndür
        if '<img src="' in response:
            start_idx = response.find('<img src="') + len('<img src="')
            end_idx = response.find('"', start_idx)
            image_url = response[start_idx:end_idx]
            
            base64_image = fetch_image_as_base64(image_url)
            if base64_image:
                # Resmi base64 formatında döndür
                response = response.replace(image_url, base64_image)

        print(f"Chatbot response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': 'An error occurred while processing your request.'}), 500

@app.route('/test', methods=['GET'])
def test():
    return "Flask çalışıyor!"

if __name__ == '__main__':
    app.run(debug=True)
