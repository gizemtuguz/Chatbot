#daha fazla bilgiye ihtiyacınız olması durumunda data_clean_epaticom.py dosyasına bakabilirsiniz


import json
from collections import defaultdict

with open('raw_data_kitaplık.json', 'r', encoding='utf-8') as json_file:
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
                if not (paragraph.startswith('Kılavuzlar Yapılandırma Örnekleri Terimler Sözlüğü ePati Siber Güvenlik Teknolojileri San. ve Tic. A.Ş. Mersin Üniversitesi Çiftlikköy KampüsüTeknopark İdari Binası Kat:4 No: 411Posta Kodu: 33343Yenişehir / Mersin / TÜRKİYE Web: www.epati.com.tre-Posta: bilgi@epati.com.trTel: +90 324 361 02 33Faks: +90 324 361 02 39')):
                    combined_data[title].append({'Paragraph': paragraph})
            if 'Image' in content:
                img_url = content['Image']
                # İstenmeyen URL'lerle başlayan img URL'lerini atlama(kullanıcın ihtiyacı olmayan resimler logo gibi)
                if not (img_url.startswith('https://kitaplik.epati.com.tr/img')):
                    combined_data[title].append({'Image': img_url})
            if 'Table' in content:
                combined_data[title].append({'Table': content['Table']})

cleaned_combined_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('cleaned_data_kitaplık.json', 'w', encoding='utf-8') as json_file:
    json.dump(cleaned_combined_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla cleaned_data_kitaplık.json dosyasına kaydedildi.')
