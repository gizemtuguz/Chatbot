#daha fazla bilgiye ihtiyacınız olması durumunda data_clean_epaticom.py dosyasına bakabilirsiniz

import json
from collections import defaultdict

with open('raw_data_kitaplık_eng.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

combined_data = defaultdict(list)

for page in data:
    for content in page['Content']:
        main_heading = content.get('Main Heading', '')
        subheading = content.get('Subheading', '')

        if main_heading or subheading:
            title_parts = []
            if main_heading:
                title_parts.append(main_heading)
            if subheading:
                title_parts.append(subheading)
            title = ' - '.join(title_parts)
            
            if 'Paragraph' in content:
                paragraph=content['Paragraph']
                if not (paragraph.startswith('Guides Configuration Examples Glossary of Terms ePati Cyber Security Co. Mersin Üniversitesi Çiftlikköy KampüsüTeknopark İdari Binası Kat:4 No: 411Posta Kodu: 33343Yenişehir / Mersin / TURKEY Web: www.epati.com.tre-Mail: info@epati.com.trTel: +90 324 361 02 33Fax: +90 324 361 02 39')):
                    combined_data[title].append({'Paragraph': paragraph})
            if 'Image' in content:
                img_url = content['Image']
                # İstenmeyen URL'lerle başlayan img URL'leri atlar (logo gibi kullanıcıya lazım olmayan resimler)
                if not (img_url.startswith('https://kb.epati.com.tr/img/')):
                    combined_data[title].append({'Image': img_url})
            if 'Table' in content:
                combined_data[title].append({'Table': content['Table']})

cleaned_combined_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('cleaned_data_kitaplık_eng.json', 'w', encoding='utf-8') as json_file:
    json.dump(cleaned_combined_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla cleaned_data_kitaplık_eng.json dosyasına kaydedildi.')
