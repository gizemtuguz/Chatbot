import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

starting_urls = [
    'https://kitaplik.epati.com.tr/sss/sss/',
    'https://kitaplik.epati.com.tr/kilavuzlar/antikor-v2-yeni-nesil-guvenlik-duvari/',
    'https://kitaplik.epati.com.tr/kilavuzlar/antikor-v2-layer2-tunelleme/',
    'https://kitaplik.epati.com.tr/kilavuzlar/antikor-v2-tunel-omurga/',
    'https://kitaplik.epati.com.tr/yapilandirma-ornekleri/antikor-v2-yeni-nesil-guvenlik-duvari/',
    'https://kitaplik.epati.com.tr/terimler-sozlugu/',
    'https://kitaplik.epati.com.tr/yapilandirma-ornekleri/antikor-v2-layer2-tunelleme/',
    'https://kitaplik.epati.com.tr/yapilandirma-ornekleri/antikor-v2-tunel-omurga/'
]

data = []

def process_strong_tags(tag):
    """
    Bu fonksiyon, bir cümlenin içinde yer alan strong etiketleri için
    başına ve sonuna boşluk ekler, ancak yalnız başına olan strong etiketleri için eklemez.
    """
    if isinstance(tag.previous_sibling, str) or isinstance(tag.next_sibling, str):
        return f" {tag.get_text(strip=True)} "
    else:
        return tag.get_text(strip=True)

def parse_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find('title').get_text(strip=True) if soup.find('title') else 'No Title'

    page_data = {"URL": url, "Title": title, "Content": []}
    elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'table', 'img', 'ul', 'blockquote'])
    current_heading_h1 = None
    current_heading_h2 = None
    current_heading_h3 = None
    current_heading_h4 = None
    paragraph_content = ""
    skip_next_p = False  # network-şeması ve hemen sonraki p'yi atlamak için
    
    for element in elements:
        if element.name == 'h1':
            if element.get('id') == 'network-şeması':
                skip_next_p = True  # Bir sonraki p etiketini atlamak için bayrak ayarlayın
                continue
            else:
                if paragraph_content:
                    content_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Paragraph': paragraph_content.strip()}
                    page_data["Content"].append(content_data)
                    paragraph_content = ""
                current_heading_h1 = element.get_text(strip=True)
                current_heading_h2 = None
                current_heading_h3 = None
                current_heading_h4 = None
        elif element.name == 'h2' or element.name == 'h4':  # h4'ü de h2 olarak kabul ediyoruz
            # Eğer herhangi bir etiketin id="network-şeması" ise ve img varsa bu etiketi ve hemen sonraki p'yi atla
            if element.get('id') == 'network-şeması':
                skip_next_p = True  # Bir sonraki p etiketini atlamak için bayrak ayarlayın
                continue
            if element.find('img'):
                paragraph_content += " " + element.get_text(strip=True)
            else:
                if paragraph_content:
                    content_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Paragraph': paragraph_content.strip()}
                    page_data["Content"].append(content_data)
                    paragraph_content = ""
                current_heading_h2 = element.get_text(strip=True)
                current_heading_h3 = None
                current_heading_h4 = None
        elif element.name == 'h3':
            if element.get('id') == 'network-şeması':
                skip_next_p = True  # Bir sonraki p etiketini atlamak için bayrak ayarlayın
                continue
            else:
                if paragraph_content:
                    content_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Paragraph': paragraph_content.strip()}
                    page_data["Content"].append(content_data)
                    paragraph_content = ""
                current_heading_h3 = element.get_text(strip=True)
                current_heading_h4 = None
        elif element.name == 'h5':
            if paragraph_content:
                content_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Paragraph': paragraph_content.strip()}
                page_data["Content"].append(content_data)
                paragraph_content = ""
        elif element.name == 'p':
            if skip_next_p:  # Eğer bir önceki etiket network-şeması id'sine sahipse bu p'yi atla
                skip_next_p = False
                continue
            # p etiketi içinde strong varsa bu fonksiyonu kullanarak işleme
            for strong_tag in element.find_all('strong'):
                strong_tag.insert_before(process_strong_tags(strong_tag))
                strong_tag.extract()  # Orijinal strong etiketini kaldır
            paragraph_content += " " + element.get_text(strip=True)
            content_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Paragraph': paragraph_content.strip()}
            page_data["Content"].append(content_data)
            paragraph_content = ""
        elif element.name == 'table':
            if paragraph_content:
                content_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Paragraph': paragraph_content.strip()}
                page_data["Content"].append(content_data)
                paragraph_content = ""
            headers = [header.get_text(strip=True) for header in element.find_all('th')]
            rows = []
            for row in element.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
                if cells:
                    rows.append(cells)
            table_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Table': {'Headers': headers, 'Rows': rows}}
            page_data["Content"].append(table_data)
        elif element.name == 'img':
            img_src = element.get('src')
            if img_src:
                img_full_url = urljoin(url, img_src)
                img_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Image': img_full_url}
                page_data["Content"].append(img_data)
        elif element.name == 'ul':
            list_items = []
            list_heading = None  # Liste için başlık eklemek için
            for index, li in enumerate(element.find_all('li'), start=1):
                list_items.append({'row': li.get_text(strip=True)})
                # Eğer li etiketinden hemen sonra strong varsa, bu liste başlığı olarak kabul edilir
                next_sibling = li.find_next_sibling()
                if next_sibling and next_sibling.name == 'strong':
                    list_heading = " " + next_sibling.get_text(strip=True) + " "  # Başına ve sonuna boşluk ekle
            if list_items:
                list_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'List': list_items}
                if list_heading:
                    list_data['List Heading'] = list_heading
                page_data["Content"].append(list_data)
        elif element.name == 'blockquote':
            # Blockquote içindeki p ve liste etiketlerini kontrol et
            p_tags = element.find_all('p')
            list_tags = element.find_all(['ul', 'ol'])
            if not p_tags and not list_tags:  # Eğer blockquote içinde p ve liste etiketi yoksa, blockquote'u işle
                blockquote_text = element.get_text(strip=True)
                if blockquote_text.strip():
                    blockquote_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Blockquote': blockquote_text.strip()}
                    page_data["Content"].append(blockquote_data)

    # If there's remaining paragraph content, append it as a final paragraph
    if paragraph_content:
        content_data = {'Main Heading': current_heading_h1, 'Subheading': current_heading_h2 or current_heading_h3 or current_heading_h4, 'Paragraph': paragraph_content.strip()}
        page_data["Content"].append(content_data)

    if page_data["Content"]:
        data.append(page_data)

    # Find all links within the current page
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(url, href)

        # Ensure that the found URL is a subpage of the starting URL
        if full_url.startswith(url) and full_url not in visited:
            to_visit.append(full_url)

# Loop through each starting URL
for starting_url in starting_urls:
    visited = set()
    to_visit = [starting_url]

    while to_visit:
        current_url = to_visit.pop(0)
        if current_url not in visited:
            print(f'Visiting: {current_url}')
            parse_page(current_url)
            visited.add(current_url)

with open('./raw_data_kitaplık.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla raw_data_kitaplık.json dosyasına kaydedildi.')
