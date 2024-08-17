import json
from collections import defaultdict
import re

# JSON dosyasını okuma
with open('cleaned_data_kitaplık_eng.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Temizlenmiş veri listesi
combined_data = defaultdict(list)

# Silinmesi istenen içerikler
unwanted_paragraph_parts = [
    "ePati Cyber Security Co. Mersin Üniversitesi Çiftlikköy KampüsüTeknopark İdari Binası Kat:4 No: 411Posta Kodu: 33343Yenişehir / Mersin / TURKEY Web: www.epati.com.tre-Mail: info@epati.com.trTel: +90 324 361 02 33Fax: +90 324 361 02 39",
    "Unwanted Paragraph 1",     
    "Unwanted Paragraph 2",             
    "Unwanted Paragraph 3"
]

# Atlanacak başlıklar (alt içerilerinde gerekli bilgiler yok)
unwanted_titles = [
    "KnowledgeBase", "Guides", "Configuration Examples", "Glossary of Terms","Glossary ofTerms",
    "Antikor v2 - Layer2 Tunnel BackBoneGuides", "Antikor v2 - Next Generation FirewallGuides",
    "Antikor v2 - Layer2 Tunnel BackBoneConfiguration Examples", "Antikor v2 - Next Generation FirewallConfiguration Examples"
]

# Başlıkları dönüştürmek için fonksiyon (terimler sözlüğü ve sık sorulan sorular için)
def transform_heading(heading):
    if heading and 'Terms Beginning with' in heading:
        parts = heading.split(' - ')
        if len(parts) > 1:
            term = parts[-1].strip()
            return f"What is the meaning of {term} ?"    
    if heading.startswith("Frequenty Asked Questions -"):
        return heading.replace("Frequenty Asked Questions -", "").strip()
    return heading

# Aynı ana başlığa sahip olup adımlara bölünmüş başlıkları gruplayacak fonksiyon
def get_main_title(heading):
    match = re.match(r"^(.*?)(?: - Step \d+)?$", heading)
    if match:
        return match.group(1).strip()
    return heading

# Parantez içindeki metni ayırma fonksiyonu
def split_title_with_parentheses(title):
    match = re.match(r"(.+?) \((.+?)\)", title)
    if match:
        return [f"{match.group(1).strip()}", f"{match.group(2).strip()}"]
    return [title]

# İçeriği birleştir ve filtreler
for item in data:
    title = item.get("Title", "")
    
    # İstenmeyen başlıkları atlar
    if title in unwanted_titles:
        continue
    
    content_list = item.get("Content", [])
    
    filtered_content = []
    
    for content in content_list:
        # İstenmeyen paragraf içeriklerini kontrol eder
        if 'Paragraph' in content and any(unwanted_part in content['Paragraph'] for unwanted_part in unwanted_paragraph_parts):
            continue  # Eğer istenmeyen içerikse atlar
        filtered_content.append(content)
    
    # Eğer filtrelenmiş içerik varsa ve içerik boş değilse, başlığı dönüştürerek yeni veriye ekle
    if filtered_content:
        split_titles = split_title_with_parentheses(title)
        for split_title in split_titles:
            transformed_title = transform_heading(split_title)
            main_title = get_main_title(transformed_title)
            combined_data[main_title].extend(filtered_content)

# Filtrelenmiş ve birleştirilmiş verileri JSON dosyasına kaydeder
final_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('filtered_data_kitaplık_eng.json', 'w', encoding='utf-8') as json_file:
    json.dump(final_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla filtered_data_kitaplık_eng.json dosyasına kaydedildi.')
