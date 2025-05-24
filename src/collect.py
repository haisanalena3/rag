import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import traceback
from config import load_config

class WebScraper:
    def __init__(self, save_dir="db"):
        # Xác định đường dẫn db
        current_dir = Path(__file__).parent
        if current_dir.name == "src":
            # Nếu đang ở trong thư mục src, lùi về thư mục cha
            project_root = current_dir.parent
            self.save_dir = project_root / save_dir
        else:
            # Nếu đang ở thư mục gốc
            self.save_dir = Path(save_dir)
            
        self.save_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.success_count = 0
        self.error_count = 0

    def is_valid_image_url(self, url):
        """Kiểm tra URL ảnh hợp lệ"""
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        invalid_patterns = ['data:', 'javascript:', 'mailto:', '#', 'tel:']
        if any(pattern in url.lower() for pattern in invalid_patterns):
            return False
            
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico']
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        if '.' in path:
            ext = os.path.splitext(path)[1]
            if ext and ext not in valid_extensions:
                return False
                
        return True

    def clean_filename(self, url):
        """Tạo tên file sạch từ URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.strip('/').replace('/', '_')
        
        if path:
            filename = f"{domain}_{path}"
        else:
            filename = domain
            
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
            
        filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
        return filename[:100]

    def extract_metadata(self, soup, url):
        """Trích xuất metadata từ trang web"""
        metadata = {
            'url': url,
            'title': '',
            'description': '',
            'keywords': '',
            'author': '',
            'language': '',
            'scraped_at': datetime.now().isoformat(),
            'image_count': 0
        }
        
        # Lấy title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
            
        # Lấy meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'}) or \
                   soup.find('meta', attrs={'property': 'og:description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '').strip()
            
        # Lấy meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            metadata['keywords'] = keywords_tag.get('content', '').strip()
            
        # Lấy author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag:
            metadata['author'] = author_tag.get('content', '').strip()
            
        # Lấy language
        lang_tag = soup.find('html')
        if lang_tag and lang_tag.get('lang'):
            metadata['language'] = lang_tag.get('lang')
            
        if not metadata['title']:
            metadata['title'] = f"Trang web từ {urlparse(url).netloc}"
            
        return metadata

    def clean_text(self, soup):
        """Làm sạch và trích xuất text từ HTML"""
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
            
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.main-content',
            '.post-content', '.entry-content', '.article-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
                
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            body = soup.find('body')
            if body:
                text = body.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 2:
                lines.append(line)
                
        return '\n'.join(lines)

    def get_image_info(self, img_tag, base_url):
        """Trích xuất thông tin ảnh từ img tag"""
        img_info = {}
        
        img_url = (img_tag.get('src') or 
                  img_tag.get('data-src') or 
                  img_tag.get('data-original') or
                  img_tag.get('data-lazy-src') or
                  img_tag.get('data-srcset'))
        
        if not img_url:
            return None
            
        img_url = urljoin(base_url, img_url)
        
        if not self.is_valid_image_url(img_url):
            return None
            
        img_info['url'] = img_url
        img_info['alt'] = img_tag.get('alt', '').strip()
        img_info['title'] = img_tag.get('title', '').strip()
        img_info['width'] = img_tag.get('width', '')
        img_info['height'] = img_tag.get('height', '')
        
        srcset = img_tag.get('srcset') or img_tag.get('data-srcset')
        if srcset:
            srcset_urls = []
            for item in srcset.split(','):
                parts = item.strip().split()
                if parts:
                    url_candidate = urljoin(base_url, parts[0])
                    if self.is_valid_image_url(url_candidate):
                        srcset_urls.append(url_candidate)
            if srcset_urls:
                img_info['url'] = srcset_urls[-1]
                
        return img_info

    def download_image(self, img_url, img_dir, filename_prefix, index):
        """Tải và lưu ảnh"""
        try:
            print(f"    Đang tải ảnh {index + 1}: {img_url}")
            
            response = self.session.get(img_url, timeout=15, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                print(f"    ⚠️ Không phải file ảnh: {content_type}")
                return None
                
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:
                print(f"    ⚠️ File quá lớn: {content_length} bytes")
                return None
                
            content = response.content
            if len(content) < 100:
                print(f"    ⚠️ File quá nhỏ: {len(content)} bytes")
                return None
                
            img_hash = hashlib.md5(content).hexdigest()[:10]
            
            ext_map = {
                'image/jpeg': '.jpg',
                'image/jpg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/webp': '.webp',
                'image/bmp': '.bmp',
                'image/svg+xml': '.svg',
                'image/x-icon': '.ico',
                'image/vnd.microsoft.icon': '.ico'
            }
            
            ext = ext_map.get(content_type)
            if not ext:
                parsed = urlparse(img_url)
                ext = os.path.splitext(parsed.path)[-1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico']:
                    ext = '.jpg'
                    
            filename = f"{filename_prefix}_img_{index}_{img_hash}{ext}"
            img_path = img_dir / filename
            
            if img_path.exists():
                print(f"    ✅ Ảnh đã tồn tại: {filename}")
                return str(img_path)
                
            try:
                if ext not in ['.svg', '.ico']:
                    img = Image.open(io.BytesIO(content))
                    
                    width, height = img.size
                    if width < 50 or height < 50:
                        print(f"    ⚠️ Ảnh quá nhỏ: {width}x{height}")
                        return None
                        
                    img.verify()
                    
                    img = Image.open(io.BytesIO(content))
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(img_path, quality=85, optimize=True)
                else:
                    with open(img_path, 'wb') as f:
                        f.write(content)
                        
                print(f"    ✅ Đã lưu: {filename}")
                return str(img_path)
                
            except Exception as e:
                try:
                    with open(img_path, 'wb') as f:
                        f.write(content)
                    print(f"    ✅ Đã lưu (raw): {filename}")
                    return str(img_path)
                except Exception as e2:
                    print(f"    ❌ Lỗi lưu file: {e2}")
                    return None
                
        except Exception as e:
            print(f"    ❌ Lỗi tải ảnh: {e}")
            return None

    def fetch_and_save(self, url):
        """Tải và lưu nội dung từ URL"""
        try:
            print(f"\n🌐 Đang xử lý: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('text/html'):
                print(f"⚠️ Không phải trang HTML: {content_type}")
                return False
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            base_filename = self.clean_filename(url)
            page_dir = self.save_dir / base_filename
            page_dir.mkdir(exist_ok=True)
            
            print(f"📁 Tạo thư mục: {page_dir.name}")
            
            metadata = self.extract_metadata(soup, url)
            print(f"📋 Title: {metadata['title']}")
            
            clean_text = self.clean_text(soup)
            if not clean_text.strip():
                print("⚠️ Không tìm thấy nội dung text")
                clean_text = f"Nội dung từ {url}\nKhông thể trích xuất text từ trang này."
                
            text_file = page_dir / "content.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print(f"💾 Lưu text: {len(clean_text)} ký tự")
                
            img_tags = soup.find_all("img")
            img_dir = page_dir / "images"
            img_dir.mkdir(exist_ok=True)
            
            print(f"🖼️ Tìm thấy {len(img_tags)} img tags")
            
            downloaded_images = []
            unique_urls = set()
            
            for idx, img_tag in enumerate(img_tags):
                img_info = self.get_image_info(img_tag, url)
                if not img_info or img_info['url'] in unique_urls:
                    continue
                    
                unique_urls.add(img_info['url'])
                
                img_path = self.download_image(
                    img_info['url'], 
                    img_dir, 
                    base_filename, 
                    idx
                )
                
                if img_path:
                    img_info['local_path'] = img_path
                    downloaded_images.append(img_info)
                    
                time.sleep(0.2)
                
            metadata['image_count'] = len(downloaded_images)
            
            required_fields = ['title', 'description', 'url', 'scraped_at']
            for field in required_fields:
                if field not in metadata or not metadata[field]:
                    if field == 'title':
                        metadata[field] = f"Trang web từ {urlparse(url).netloc}"
                    elif field == 'description':
                        metadata[field] = "Không có mô tả"
                    elif field == 'url':
                        metadata[field] = url
                    elif field == 'scraped_at':
                        metadata[field] = datetime.now().isoformat()
            
            metadata_file = page_dir / "metadata.json"
            metadata_content = {
                'metadata': metadata,
                'images': downloaded_images
            }
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_content, f, ensure_ascii=False, indent=2)
            
            if not metadata_file.exists():
                print("❌ Lỗi: metadata.json không được tạo")
                return False
                
            print(f"💾 Lưu metadata: {metadata_file.name}")
            print(f"✅ Hoàn thành {url}")
            print(f"   📄 Text: {text_file.name}")
            print(f"   🖼️ Ảnh: {len(downloaded_images)} files")
            print(f"   📋 Metadata: {metadata_file.name}")
            
            self.success_count += 1
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Lỗi kết nối {url}: {e}")
            self.error_count += 1
            return False
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {url}: {e}")
            print("Chi tiết lỗi:")
            traceback.print_exc()
            self.error_count += 1
            return False

    def print_summary(self):
        """In tóm tắt kết quả"""
        print(f"\n{'='*50}")
        print(f"📊 TÓM TẮT KẾT QUẢ")
        print(f"{'='*50}")
        print(f"✅ Thành công: {self.success_count}")
        print(f"❌ Lỗi: {self.error_count}")
        print(f"📁 Dữ liệu lưu tại: {self.save_dir.absolute()}")
        
        if self.save_dir.exists():
            subdirs = [d for d in self.save_dir.iterdir() if d.is_dir()]
            total_images = 0
            for subdir in subdirs:
                img_dir = subdir / "images"
                if img_dir.exists():
                    total_images += len(list(img_dir.glob("*")))
            print(f"🖼️ Tổng số ảnh: {total_images}")
        print(f"{'='*50}")

def main():
    """Hàm chính"""
    try:
        print("🚀 Bắt đầu thu thập dữ liệu...")
        
        config = load_config()
        urls = config.get("URLS", [])
        
        if not urls:
            print("❌ Không tìm thấy URLs trong config!")
            print("Vui lòng kiểm tra file config.py và đảm bảo có danh sách URLS")
            return
            
        print(f"📋 Tìm thấy {len(urls)} URLs trong config")
        
        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")
        
        scraper = WebScraper()
        
        print(f"\n🔄 Bắt đầu thu thập dữ liệu từ {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Xử lý URL: {url}")
            
            success = scraper.fetch_and_save(url)
            
            if i < len(urls):
                print("⏳ Chờ 2 giây trước khi xử lý URL tiếp theo...")
                time.sleep(2)
        
        scraper.print_summary()
        
        if scraper.success_count > 0:
            print(f"\n🎉 Hoàn thành thu thập dữ liệu!")
            print(f"Bạn có thể chạy ứng dụng Streamlit để xem kết quả:")
            print(f"streamlit run app.py")
        else:
            print(f"\n😞 Không thu thập được dữ liệu nào!")
            print(f"Vui lòng kiểm tra:")
            print(f"- Kết nối internet")
            print(f"- URLs trong config.py có hợp lệ không")
            print(f"- Các website có thể truy cập được không")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình thực thi: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
