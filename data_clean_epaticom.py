import json
from collections import defaultdict

#json dosyasını okuma
with open('raw_data_epaticom.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

#verileri başlığa göre gruplandırır ve birleştirir
combined_data = defaultdict(list)

for page in data:
    for content in page['Content']:
        #main heading ve subheading in boş olup olmadığını kontrol eder
        main_heading = content.get('Main Heading', '')
        subheading = content.get('Subheading', '')

        if main_heading or subheading:
            #main Heading ve subheadingi birleştirip title yapma
            title_parts = []
            if main_heading:
                title_parts.append(main_heading)
            if subheading:
                title_parts.append(subheading)
            title = ' - '.join(title_parts)         #main ve subheading arasına " - " işareti koyarak başlığı günceller
            
            #içeriğin sırasını koruyarak başlığın altına ekler
            if 'Paragraph' in content:
                paragraph = content['Paragraph']
                combined_data[title].append({'Paragraph': paragraph})
            if 'Image' in content:
                img_url = content['Image']
                combined_data[title].append({'Image': img_url})
            if 'Table' in content:
                combined_data[title].append({'Table': content['Table']})

#birleştirilmiş ve temizlenmiş veriyi dataframe kaydetme
cleaned_combined_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('cleaned_data_epaticom.json', 'w', encoding='utf-8') as json_file:
    json.dump(cleaned_combined_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla cleaned_data_epaticom.json dosyasına kaydedildi.')
