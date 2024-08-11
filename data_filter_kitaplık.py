import json
from collections import defaultdict
import re

# JSON dosyasını okur
with open('cleaned_data_kitaplık.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

combined_data = defaultdict(list)

# Silinmek istenen içerikler
unwanted_paragraph_parts = [
    "ePati Siber Güvenlik Teknolojileri San. ve Tic. A.Ş. Mersin Üniversitesi Çiftlikköy KampüsüTeknopark İdari Binası Kat:4 No: 411Posta Kodu: 33343Yenişehir / Mersin / TÜRKİYE Web: www.epati.com.tre-Posta: bilgi@epati.com.trTel: +90 324 361 02 33Faks: +90 324 361 02 39",
    "Kılavuz",
    "Yapılandırma Örnekleri",
    "İstenmeyen Paragraf 3"
]

# Atlanacak başlıklar
unwanted_titles = [
    "Kitaplık", "Kılavuzlar", "Yapılandırma Örnekleri", "Terimler Sözlüğü","TerimlerSözlüğü",
    "Antikor v2 - Yeni Nesil Güvenlik DuvarıKılavuzlar", "Antikor v2 - Layer2 Tünel OmurgaKılavuzlar",
    "Antikor v2 - Yeni Nesil Güvenlik DuvarıYapılandırma Örnekleri", "Antikor v2 - Yeni Nesil Güvenlik DuvarıYapılandırma Örnekleri"
]

# Başlığı kontrol edip atlama fonksiyonu
def should_skip_title(title):
    return any(unwanted_title in title for unwanted_title in unwanted_titles)

# Adım bazlı başlıkları birleştirmek için fonksiyon
def merge_step_titles(title):
    match = re.match(r"(.+?) - Adım \d+", title)
    if match:
        return match.group(1).strip()
    return title

# Başlıkları dönüştürmek için fonksiyon
def transform_heading(heading):
    if heading and 'ile Başlayan Terimler' in heading:
        parts = heading.split(' - ')
        if len(parts) > 1:
            term = parts[-1].strip()
            return f"{term} ne demek?"
    if heading.startswith("Sık Sorulan Sorular -"):
        return heading.replace("Sık Sorulan Sorular -", "").strip()
    return heading

# İçeriği birleştir ve filtrele
for item in data:
    title = item.get("Title", "")
    
    # Başlığı atlama kontrolü
    if should_skip_title(title):
        continue
    
    content_list = item.get("Content", [])
    
    # Başlıkları Adım numaralarına göre birleştir
    merged_title = merge_step_titles(title)
    filtered_content = []
    
    for content in content_list:
        # İstenmeyen paragraf içeriklerini kontrol et
        if 'Paragraph' in content and any(unwanted_part in content['Paragraph'] for unwanted_part in unwanted_paragraph_parts):
            continue  # İstenmeyen içeriği atla
        filtered_content.append(content)
    
    # Eğer filtrelenmiş içerik varsa ve içerik boş değilse, başlığı dönüştürerek yeni veriye ekle
    if filtered_content:
        transformed_title = transform_heading(merged_title)
        combined_data[transformed_title].extend(filtered_content)

# Filtrelenmiş ve birleştirilmiş verileri JSON dosyasına kaydetme
final_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('filtered_data_kitaplık.json', 'w', encoding='utf-8') as json_file:
    json.dump(final_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla filtered_data_kitaplık.json dosyasına kaydedildi.')
