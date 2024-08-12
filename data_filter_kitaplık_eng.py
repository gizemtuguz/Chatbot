import json
from collections import defaultdict
import re

#json dosyasını okuma
with open('cleaned_data_kitaplık_eng.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

#temizlenmiş veri listesi
combined_data = defaultdict(list)

#silinmesi istenen içerikler
unwanted_paragraph_parts = [
    "ePati Cyber Security Co. Mersin Üniversitesi Çiftlikköy KampüsüTeknopark İdari Binası Kat:4 No: 411Posta Kodu: 33343Yenişehir / Mersin / TURKEY Web: www.epati.com.tre-Mail: info@epati.com.trTel: +90 324 361 02 33Fax: +90 324 361 02 39",
    "Unwanted Paragraph 1",     
    "Unwanted Paragraph 2",             #daha sonra kullanılmak üzere boş alanlar bırakıldı
    "Unwanted Paragraph 3"
]

#atlanacak başlıklar (alt içerilerinde gerekli bilgiler yok)
unwanted_titles = [
    "KnowledgeBase", "Guides", "Configuration Examples", "Glossary of Terms","Glossary ofTerms",
    "Antikor v2 - Layer2 Tunnel BackBoneGuides", "Antikor v2 - Next Generation FirewallGuides",
    "Antikor v2 - Layer2 Tunnel BackBoneConfiguration Examples", "Antikor v2 - Next Generation FirewallConfiguration Examples"
]

#başlıkları dönüştürmek için fonksiyon (terimelr sözlüğü ve sık sorulan sorular için)
def transform_heading(heading):
    if heading and 'Terms Beginning with' in heading:
        parts = heading.split(' - ')
        if len(parts) > 1:
            term = parts[-1].strip()
            return f"What is the meaning of {term} ?"    
    if heading.startswith("Frequenty Asked Questions -"):
        return heading.replace("Frequenty Asked Questions -", "").strip()
    return heading

#aynı ana başlığa sahip olup adımlara bölünmüş başlıkları gruplayacak fonksiyon
def get_main_title(heading):
    match = re.match(r"^(.*?)(?: - Step \d+)?$", heading)
    if match:
        return match.group(1).strip()
    return heading

#içeriği birleştir ve filtreler
for item in data:
    title = item.get("Title", "")
    
    #istenmeyen başlıkları atlar
    if title in unwanted_titles:
        continue
    
    content_list = item.get("Content", [])
    
    filtered_content = []
    
    for content in content_list:
        #istenmeyen paragraf içeriklerini kontrol eder
        if 'Paragraph' in content and any(unwanted_part in content['Paragraph'] for unwanted_part in unwanted_paragraph_parts):
            continue  #eğer istenmeyen içerikse atlar
        filtered_content.append(content)
    
    #eğer filtrelenmiş içerik varsa ve içerik boş değilse, başlığı dönüştürerek yeni veriye ekle
    if filtered_content:
        transformed_title = transform_heading(title)
        main_title = get_main_title(transformed_title)
        combined_data[main_title].extend(filtered_content)

#filtrelenmiş ve birleştirilmiş verileri JSON dosyasına kaydeder
final_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('filtered_data_kitaplık_eng.json', 'w', encoding='utf-8') as json_file:
    json.dump(final_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla filtered_data_kitaplık_eng.json dosyasına kaydedildi.')
