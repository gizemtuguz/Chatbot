import json
from collections import defaultdict

#json dosyasını okuma
with open('cleaned_data_epaticom.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

#temizlenmiş veri listesi
combined_data = defaultdict(list)

#içeriği birleştir ve ekle
for item in data:
    title = item.get("Title", "")
    content_list = item.get("Content", [])
    
    #içeriği birleştirme
    combined_data[title].extend(content_list)

#birleştirilmiş verileri JSON dosyasına kaydetme
final_data = [{'Title': title, 'Content': contents} for title, contents in combined_data.items()]

with open('filtered_data_epaticom.json', 'w', encoding='utf-8') as json_file:
    json.dump(final_data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla filtered_data_epaticom.json dosyasına kaydedildi.')
