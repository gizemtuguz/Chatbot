import json
from collections import defaultdict

# JSON dosyasını okuma
with open('raw_data_epaticom.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Verileri başlığa göre gruplama ve birleştirme
combined_data = defaultdict(list)

for page in data:
    for content in page['Content']:
        # Main Heading ve Subheading'in boş olup olmadığını kontrol et
        main_heading = content.get('Main Heading', '')
        subheading = content.get('Subheading', '')

        if main_heading or subheading:
            # Main Heading ve Subheading'i birleştirip Title yapma
            title_parts = []
            if main_heading:
                title_parts.append(main_heading)
            if subheading:
                title_parts.append(subheading)
            title = ' - '.join(title_parts)
            
            # İçeriği sırasını koruyarak ekleme
            if 'Paragraph' in content:
                paragraph = content['Paragraph']
                combined_data[title].append({'Paragraph': paragraph})
            if 'Image' in content:
                img_url = content['Image']
                combined_data[title].append({'Image': img_url})
            if 'Table' in content:
                combined_data[title].append({'Table': content['Table']})

# Birleştirilmiş verileri JSON dosyasına kaydetme
cleaned_combined_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('cleaned_data_epaticom.json', 'w', encoding='utf-8') as json_file:
    json.dump(cleaned_combined_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla cleaned_data_epaticom.json dosyasına kaydedildi.')
