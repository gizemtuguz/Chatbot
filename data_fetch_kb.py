import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import time

starting_url = 'https://kb.epati.com.tr/'
visited = set()
to_visit = [starting_url]

data = []

def parse_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find('title').get_text(strip=True) if soup.find('title') else 'No Title'

    page_data = {"URL": url, "Title": title, "Content": []}
    elements = soup.find_all(['h1', 'h3', 'h4', 'p', 'table', 'img', 'ul'])
    current_heading = None
    current_subheading = None
    paragraph_content = ""
    
    for element in elements:
        if element.name == 'h1':
            if paragraph_content:
                content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}
                page_data["Content"].append(content_data)
                paragraph_content = ""
            current_heading = element.get_text(strip=True)
            current_subheading = None
        elif element.name == 'h3' or element.name == 'h4':
            if paragraph_content:
                content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}
                page_data["Content"].append(content_data)
                paragraph_content = ""
            current_subheading = element.get_text(strip=True)
        elif element.name == 'p':
            paragraph_content += " " + element.get_text(strip=True)
        elif element.name == 'table':
            if paragraph_content:
                content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}
                page_data["Content"].append(content_data)
                paragraph_content = ""
            headers = [header.get_text(strip=True) for header in element.find_all('th')]
            rows = []
            for row in element.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
                if cells:
                    rows.append(cells)
            table_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Table': {'Headers': headers, 'Rows': rows}}
            page_data["Content"].append(table_data)
        elif element.name == 'img':
            img_src = element.get('src')
            if img_src:
                img_full_url = urljoin(starting_url, img_src)
                img_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Image': img_full_url}
                page_data["Content"].append(img_data)
        elif element.name == 'ul':
            rows = []
            for li in element.find_all('li'):
                code_content = li.find('code').get_text(strip=True) if li.find('code') else ''
                description = li.get_text(strip=True).replace(code_content, '').strip()
                rows.append([code_content, description])
            
            if rows:
                list_as_table = {'Headers': ['Command', 'Description'], 'Rows': rows}
                list_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Table': list_as_table}
                page_data["Content"].append(list_data)

    if paragraph_content:
        content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}
        page_data["Content"].append(content_data)

    if page_data["Content"]:
        data.append(page_data)

    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(starting_url, href)

        # web sitesinin html'inde dosya yolu farklı geldiği için düzenleme
        path_parts = urlparse(full_url).path.split('/')
        if len(path_parts) == 2 and len(path_parts[1]) == 1 and path_parts[1].isalpha():
            full_url = urljoin(starting_url, f'glossary-of-terms//{path_parts[1]}/')

        if (urlparse(full_url).netloc == urlparse(starting_url).netloc and 
            full_url not in visited and 
            full_url not in to_visit):
            to_visit.append(full_url)

while to_visit:
    current_url = to_visit.pop(0)
    if current_url not in visited:
        print(f'Visiting: {current_url}')
        parse_page(current_url)
        visited.add(current_url)
        time.sleep(1)

with open('raw_data_kitaplık_eng.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla raw_data_kitaplık_eng.json dosyasına kaydedildi.')
