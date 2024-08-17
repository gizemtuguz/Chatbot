import json
import re
import pandas as pd

def expand_titles_with_abbreviations(data):
    expanded_data = []

    for entry in data:
        main_title = entry.get('Title', '')
        content = entry.get('Content', [])

        # Orijinal başlığı ekle
        expanded_data.append({"Title": main_title, "Content": content})

        # Parantez içindeki kısaltmayı ve açıklamasını yakalayacak regex deseni
        match = re.search(r'\((.*?)\)', main_title)
        if match:
            abbreviation = match.group(1)  # Parantez içindeki kısaltma
            expanded_title = re.sub(r'\s*\(.*?\)\s*', '', main_title).strip()  # Parantez içini çıkararak başlığı genişlet

            # Genişletilmiş başlığı ekle
            expanded_data.append({"Title": expanded_title, "Content": content})

            # Kısaltmayı tek başına başlık olarak ekle
            expanded_data.append({"Title": abbreviation, "Content": content})

    return expanded_data

def clean_text(text):
    # Noktalama işaretlerini temizle ve diğer temizlik işlemleri
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = re.sub(r'<.*?>', '', text)  # HTML etiketlerini kaldır
    text = re.sub(r'\s+', ' ', text)  # Fazla boşlukları tek boşluğa indir
    text = re.sub(r'\[.*?\]', '', text)  # Köşeli parantez içindeki metni kaldır
    text = text.strip()  # Başlangıç ve sonundaki boşlukları kaldır
    return text

def process_table_rows_for_training(table):
    rows = table.get('Rows', [])
    processed_rows = []

    for row in rows:
        processed_row = [cell.strip() for cell in row if cell.strip()]
        if processed_row:
            processed_rows.append(processed_row)
    
    return processed_rows

def process_list_items(list_items):
    processed_items = []
    for item in list_items:
        if isinstance(item, dict) and 'row' in item:
            processed_items.append(item['row'].strip())
    return processed_items

def process_blockquote(blockquote):
    return blockquote.strip()

def process_data(file_path, clean=False):
    question_answer_data = []
    plain_text_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        expanded_data = expand_titles_with_abbreviations(data)
        
        for entry in expanded_data:
            title = entry.get('Title', '')
            content_list = entry.get('Content', [])
            
            if not content_list:
                continue

            structured_content = []

            for content_item in content_list:
                if 'Paragraph' in content_item:
                    paragraph = content_item['Paragraph'].strip()
                    if paragraph:
                        if clean:
                            paragraph = clean_text(paragraph)
                        structured_content.append({"Paragraph": paragraph})

                if 'Table' in content_item:
                    table_content = process_table_rows_for_training(content_item['Table'])
                    if table_content:
                        structured_content.append({"Table": table_content})

                if 'List' in content_item:
                    list_content = process_list_items(content_item['List'])
                    if list_content:
                        structured_content.append({"List": list_content})

                if 'Blockquote' in content_item:
                    blockquote_content = process_blockquote(content_item['Blockquote'])
                    if blockquote_content:
                        structured_content.append({"Blockquote": blockquote_content})

            if structured_content:  # Eğer içerik boş değilse
                if "?" in title:
                    question_answer_data.append({"Title": title, "Content": structured_content})
                else:
                    plain_text_data.append({"Title": title, "Content": structured_content})

    return question_answer_data, plain_text_data

def merge_datasets(*datasets):
    merged_data = []
    for dataset in datasets:
        merged_data.extend(dataset)
    return merged_data

if __name__ == "__main__":
    # Her bir dosya için veri işle, tablo, liste, blockquote verilerini koruyarak
    processed_tr_data_qa, processed_tr_data_plain = process_data('./filtered_data_kitaplık.json', clean=False)
    processed_en_data_qa, processed_en_data_plain = process_data('./filtered_data_kitaplık_eng.json', clean=False)
    processed_basic_data_qa, processed_basic_data_plain = process_data('./basic_data.json', clean=False)
    processed_epaticom_data_qa, processed_epaticom_data_plain = process_data('./filtered_data_epaticom.json', clean=False)

    # İşlenmiş verileri birleştir
    merged_data_qa = merge_datasets(processed_tr_data_qa, processed_en_data_qa, processed_basic_data_qa, processed_epaticom_data_qa)
    merged_data_plain = merge_datasets(processed_tr_data_plain, processed_en_data_plain, processed_basic_data_plain, processed_epaticom_data_plain)

    # Verileri kaydet
    with open('data_question_answer.json', 'w', encoding='utf-8') as f:
        json.dump(merged_data_qa, f, ensure_ascii=False, indent=4)

    with open('data_plain_text.json', 'w', encoding='utf-8') as f:
        json.dump(merged_data_plain, f, ensure_ascii=False, indent=4)

    # Pandas DataFrame'e çevir ve tabloyu göster
    df_qa = pd.DataFrame(merged_data_qa)
    df_plain = pd.DataFrame(merged_data_plain)

    print("Question-Answer Data (İlk 5 Satır):")
    print(df_qa.head())

    print("\nPlain Text Data (İlk 5 Satır):")
    print(df_plain.head())
