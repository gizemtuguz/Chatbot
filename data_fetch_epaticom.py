import requests         # web sitesinden veri çekme kütüphanesi
from bs4 import BeautifulSoup       #HTML içeriklerini parçalama 
import json
from urllib.parse import urljoin, urlparse      #urlleri birleştirme ve cözümleme
import time

# Web sitesinin başlangıç URL'si
starting_url = 'https://www.epati.com.tr/'          #başlangıç urlsi
visited = set()                                 #daha önce ziyaret edilen urlleri tutarak bir daha ziyareti önlemek için
to_visit = [starting_url]                           #ziyaret edilmesi gereken urllerin olduğu liste

data = []               #url içeriklerini saklama
seen_contents = set()   #daha önce toplanmış veriler

#hariç tutulan urller ve alt urlleri
excluded_urls = [
    'https://www.epati.com.tr/tr/kurumsal/etkinlikler/',
    'https://www.epati.com.tr/tr/kurumsal/basinda-epati/',
    'https://www.epati.com.tr/en/corporate/activities/',
    'https://www.epati.com.tr/en/corporate/press/'
]

#fonksiyona giren urleyi ziyaret eder içerikleri alır hata olursa hatayı basar
def parse_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return
    
    try:
        soup = BeautifulSoup(response.content, 'html5lib')      #içerik parse edilir
    except Exception as e:
        print(f"BeautifulSoup parsing error: {e}")
        return

    #sayfa başlığını alır
    title = soup.find('title').get_text(strip=True) if soup.find('title') else 'No Title'

    
    page_data = {"URL": url, "Title": title, "Content": []}     #url title contentli boş liste
    elements = soup.find_all(['h1', 'h3', 'p', 'table', 'img'])         #belirtilen elementler toplanır
    current_heading = None
    current_subheading = None
    paragraph_content = ""
    
    for element in elements:
        if element.name == 'h1':                               #ana başlık seçimi
            if paragraph_content:
                content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}
                page_data["Content"].append(content_data)
                paragraph_content = ""
            current_heading = element.get_text(strip=True)
            current_subheading = None             #önceki sub headingle karışmasına veya alt başlığı olmamasına yönelik koruma
        elif element.name == 'h3':
            if paragraph_content:
                content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}          #önceki başlıklar ve alt başlıklarla birlikte saklanır
                content_json = json.dumps(content_data, sort_keys=True)
                if content_json not in seen_contents:               #daha önce alınmamışsa veri contente eklenir
                    page_data["Content"].append(content_data)
                    seen_contents.add(content_json)
                paragraph_content = ""
            current_subheading = element.get_text(strip=True)
        elif element.name == 'p':
            if current_heading:
                paragraph_content += " " + element.get_text(strip=True)         #yeni paragraf eski paragrafa eklenir       
        elif element.name == 'table':
            if paragraph_content:
                content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}
                content_json = json.dumps(content_data, sort_keys=True)
                if content_json not in seen_contents:
                    page_data["Content"].append(content_data)
                    seen_contents.add(content_json)
                paragraph_content = ""
            headers = [header.get_text(strip=True) for header in element.find_all('th')]            #th etiketliler tablo başlıkları
            rows = []
            for row in element.find_all('tr'):                  
                cells = [cell.get_text(strip=True) for cell in row.find_all('td')]                  #tr etiketi tüm satırları td etiketi içerikleri toplar
                if cells:
                    rows.append(cells)
            table_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Table': {'Headers': headers, 'Rows': rows}}       #önceki heading, sub heading ve tablo içerikleri table datada saklanır
            table_json = json.dumps(table_data, sort_keys=True)
            if table_json not in seen_contents:
                page_data["Content"].append(table_data)         #daha önce görülmediyse ve kaydedilmediyse contente eklenir
                seen_contents.add(table_json)
        elif element.name == 'img':
            img_src = element.get('src')
            if img_src:
                img_full_url = urljoin(starting_url, img_src)                       #eğer img urlsi tam değilse "/img/asd.jpg" gibi bunu tamamlayan fonksiyon
                if img_full_url.startswith('https://www.epati.com.tr/assets/'):     #bu url ile başlayan imgler atlanır(çünkü bu url ile başlayanlar gerekli değil deneyiniz: "https://www.epati.com.tr/assets/images/siber-kume-beyaz.png")
                    continue
                img_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Image': img_full_url}       #önceki headingler ile birlikte img saklanır
                img_json = json.dumps(img_data, sort_keys=True)
                if img_json not in seen_contents:
                    page_data["Content"].append(img_data)                       #daha önce alınmadıysa contente kaydedilir
                    seen_contents.add(img_json)
    
    #kalan veri varsa onlar eklenir
    if paragraph_content:
        content_data = {'Main Heading': current_heading, 'Subheading': current_subheading, 'Paragraph': paragraph_content.strip()}
        content_json = json.dumps(content_data, sort_keys=True)
        if content_json not in seen_contents:
            page_data["Content"].append(content_data)
            seen_contents.add(content_json)

    #alınan verileri kaydeder
    if page_data["Content"]:
        data.append(page_data)

    for link in soup.find_all('a', href=True):          #tüm bağlantılar(a etiketli) toplanır
        href = link['href']
        full_url = urljoin(starting_url, href)

        #hariç tutlan urller ve pdfler atlanır
        if full_url.endswith('.pdf') or any(full_url.startswith(excluded_url) for excluded_url in excluded_urls):
            continue

        if (urlparse(full_url).netloc == urlparse(starting_url).netloc and 
            full_url not in visited and 
            full_url not in to_visit):
            to_visit.append(full_url)           #url daha önce ziyaret edilen veya edilecek listesinde yoksa ziyarte edileceğe eklenir


while to_visit:
    current_url = to_visit.pop(0)
    if current_url not in visited:
        print(f'Visiting: {current_url}')           #ziyaret edilen urller terminale yazıdırılır
        parse_page(current_url)
        visited.add(current_url)
        time.sleep(1)                               #site engellemesin diye 1 sn beklenir

#veriler jsona kaydedilir
with open('raw_data_epaticom.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print('Veriler başarıyla raw_data_epaticom.json dosyasına kaydedildi.')
