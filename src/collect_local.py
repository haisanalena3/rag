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
import shutil
import re
from config_local import load_config_local
import logging
import random

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraperLocal:
    def __init__(self, save_dir="db", overwrite=True):
        # Xác định đường dẫn db
        current_dir = Path(__file__).parent
        if current_dir.name == "src":
            project_root = current_dir.parent
            self.save_dir = project_root / save_dir
        else:
            self.save_dir = Path(save_dir)

        # Xử lý ghi đè database
        if overwrite and self.save_dir.exists():
            print(f"🗑️ Đang xóa database cũ: {self.save_dir}")
            shutil.rmtree(self.save_dir, ignore_errors=True)
            print(f"✅ Đã xóa database cũ")

        self.save_dir.mkdir(exist_ok=True)
        print(f"📁 Tạo thư mục database: {self.save_dir.absolute()}")

        # Load config
        self.config = load_config_local()

        # Cấu hình session với headers nâng cao và rotation
        self.session = requests.Session()
        self.user_agents = self.config['SCRAPING']['user_agents']
        self.update_session_headers()

        self.success_count = 0
        self.error_count = 0

    def update_session_headers(self):
        """Cập nhật headers với user agent ngẫu nhiên"""
        user_agent = random.choice(self.user_agents)
        headers = self.config['SCRAPING']['headers'].copy()
        headers['User-Agent'] = user_agent
        self.session.headers.update(headers)

    def is_valid_image_url(self, url):
        """Kiểm tra URL ảnh hợp lệ với nhiều điều kiện"""
        if not url or not url.startswith(('http://', 'https://')):
            return False

        invalid_patterns = ['data:', 'javascript:', 'mailto:', '#', 'tel:', 'blob:']
        if any(pattern in url.lower() for pattern in invalid_patterns):
            return False

        try:
            parsed = urlparse(url)
            if not parsed.netloc or parsed.netloc in ['localhost', '127.0.0.1']:
                return False
        except:
            return False

        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico', '.tiff', '.tif']
        path = parsed.path.lower()
        if '.' in path:
            ext = os.path.splitext(path)[1]
            if ext and ext not in valid_extensions:
                return False

        return True

    def clean_filename(self, url):
        """Tạo tên file sạch từ URL với xử lý tốt hơn"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.strip('/').replace('/', '_')
        
        if path:
            filename = f"{domain}_{path}"
        else:
            filename = domain

        if parsed.query:
            query_hash = hashlib.md5(parsed.query.encode()).hexdigest()[:8]
            filename += f"_{query_hash}"

        # Loại bỏ ký tự không hợp lệ
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Chỉ giữ ký tự alphanumeric và một số ký tự đặc biệt
        filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
        
        return filename[:100]  # Giới hạn độ dài

    def extract_enhanced_metadata(self, soup, url):
        """Trích xuất metadata chi tiết cho search tốt hơn"""
        metadata = {
            'url': url,
            'title': '',
            'description': '',
            'keywords': [],
            'author': '',
            'language': '',
            'scraped_at': datetime.now().isoformat(),
            'image_count': 0,
            'scraper_version': 'enhanced_v2.0',
            'content_type': '',
            'word_count': 0,
            'headings': [],
            'meta_tags': {},
            'canonical_url': '',
            'published_date': '',
            'semantic_keywords': []
        }

        # Lấy title với nhiều phương pháp
        title_sources = [
            soup.find('title'),
            soup.find('meta', attrs={'property': 'og:title'}),
            soup.find('meta', attrs={'name': 'twitter:title'}),
            soup.find('h1')
        ]

        for source in title_sources:
            if source:
                title = source.get('content') if source.name == 'meta' else source.get_text()
                if title and title.strip():
                    metadata['title'] = title.strip()
                    break

        # Lấy description từ nhiều nguồn
        desc_sources = [
            soup.find('meta', attrs={'name': 'description'}),
            soup.find('meta', attrs={'property': 'og:description'}),
            soup.find('meta', attrs={'name': 'twitter:description'})
        ]

        for source in desc_sources:
            if source and source.get('content'):
                metadata['description'] = source.get('content').strip()
                break

        # Lấy keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            keywords = [k.strip() for k in keywords_tag.get('content').split(',')]
            metadata['keywords'] = keywords

        # Extract headings với cải tiến
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        metadata['headings'] = [h.get_text(strip=True) for h in headings if h.get_text(strip=True)]

        # Tạo semantic keywords từ headings và title
        semantic_keywords = []
        for heading in metadata['headings']:
            # Trích xuất từ khóa quan trọng từ heading
            words = re.findall(r'\b\w{4,}\b', heading.lower())
            semantic_keywords.extend(words)
        
        # Thêm từ khóa từ title
        if metadata['title']:
            title_words = re.findall(r'\b\w{4,}\b', metadata['title'].lower())
            semantic_keywords.extend(title_words)
        
        metadata['semantic_keywords'] = list(set(semantic_keywords))[:20]  # Top 20 unique keywords

        # Lấy author
        author_sources = [
            soup.find('meta', attrs={'name': 'author'}),
            soup.find('meta', attrs={'property': 'article:author'}),
            soup.find('meta', attrs={'name': 'twitter:creator'})
        ]

        for source in author_sources:
            if source and source.get('content'):
                metadata['author'] = source.get('content').strip()
                break

        # Lấy language
        lang_tag = soup.find('html')
        if lang_tag and lang_tag.get('lang'):
            metadata['language'] = lang_tag.get('lang')

        # Lấy ngày xuất bản
        date_sources = [
            soup.find('meta', attrs={'property': 'article:published_time'}),
            soup.find('meta', attrs={'name': 'publish_date'}),
            soup.find('meta', attrs={'name': 'date'}),
            soup.find('time', attrs={'datetime': True})
        ]

        for source in date_sources:
            if source:
                date_val = source.get('content') or source.get('datetime')
                if date_val:
                    metadata['published_date'] = date_val.strip()
                    break

        # Xác định loại nội dung với cải tiến
        url_lower = url.lower()
        content_indicators = {
            'blog': ['blog', 'post', 'article', 'bai-viet'],
            'news': ['news', 'tin-tuc', 'thoi-su'],
            'tutorial': ['tutorial', 'guide', 'huong-dan', 'how-to', 'thuc-hanh'],
            'documentation': ['doc', 'documentation', 'api'],
            'research': ['research', 'nghien-cuu', 'study']
        }

        for content_type, indicators in content_indicators.items():
            if any(indicator in url_lower for indicator in indicators):
                metadata['content_type'] = content_type
                break

        if not metadata['content_type']:
            metadata['content_type'] = 'general'

        # Fallback title nếu không có
        if not metadata['title']:
            metadata['title'] = f"Trang web từ {urlparse(url).netloc}"

        return metadata

    def extract_semantic_content(self, soup):
        """Trích xuất content với semantic understanding cải tiến"""
        # Loại bỏ các thẻ không cần thiết
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form"]):
            element.decompose()

        # Tìm main content với nhiều selector
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.main-content',
            '.post-content', '.entry-content', '.article-content', '.article-body',
            '.post-body', '.content-body', '.text-content', '#content', '.entry',
            '.post', '.article', '.blog-content', '.page-content'
        ]

        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            # Structured content extraction với cải tiến
            headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            paragraphs = main_content.find_all(['p', 'div', 'span', 'li'])
            
            structured_sections = []
            
            # Thêm headings với format đặc biệt
            for heading in headings:
                text = heading.get_text(strip=True)
                if text and len(text) > 3:
                    structured_sections.append(f"## {text}")

            # Thêm paragraphs với filter cải tiến
            for para in paragraphs:
                text = para.get_text(strip=True)
                if text and len(text) > 20:
                    # Loại bỏ text không có ý nghĩa
                    if not re.match(r'^\d+$', text) and len(text.split()) > 3:
                        # Loại bỏ noise content
                        noise_patterns = ['cookie', 'privacy', 'terms', 'advertisement', 'ads']
                        if not any(noise in text.lower() for noise in noise_patterns):
                            structured_sections.append(text)

            final_text = '\n\n'.join(structured_sections)
        else:
            # Fallback: lấy từ body
            body = soup.find('body')
            if body:
                final_text = body.get_text(separator="\n", strip=True)
            else:
                final_text = soup.get_text(separator="\n", strip=True)

        # Làm sạch text với cải tiến
        lines = []
        for line in final_text.split('\n'):
            line = line.strip()
            if (line and len(line) > 10 and
                not re.match(r'^[^\w\s]*$', line) and  # Không phải chỉ ký tự đặc biệt
                not re.match(r'^\d+$', line) and       # Không phải chỉ số
                len(line.split()) > 2):                # Ít nhất 3 từ
                lines.append(line)

        return '\n'.join(lines)

    def get_enhanced_image_info(self, img_tag, base_url):
        """Trích xuất thông tin ảnh nâng cao với nhiều fallback"""
        # Thử nhiều attribute để lấy URL ảnh
        img_url = (img_tag.get('src') or
                   img_tag.get('data-src') or
                   img_tag.get('data-original') or
                   img_tag.get('data-lazy-src') or
                   img_tag.get('data-srcset'))

        if not img_url:
            return None

        # Xử lý srcset để lấy ảnh chất lượng cao nhất
        srcset = img_tag.get('srcset') or img_tag.get('data-srcset')
        if srcset:
            srcset_urls = []
            for item in srcset.split(','):
                parts = item.strip().split()
                if parts:
                    url_candidate = urljoin(base_url, parts[0])
                    if self.is_valid_image_url(url_candidate):
                        width = 0
                        if len(parts) > 1:
                            try:
                                width = int(parts[1].replace('w', ''))
                            except:
                                pass
                        srcset_urls.append((url_candidate, width))
            
            if srcset_urls:
                # Sắp xếp theo width và lấy ảnh lớn nhất
                srcset_urls.sort(key=lambda x: x[1], reverse=True)
                img_url = srcset_urls[0][0]

        img_url = urljoin(base_url, img_url)

        if not self.is_valid_image_url(img_url):
            return None

        img_info = {
            'url': img_url,
            'alt': img_tag.get('alt', '').strip(),
            'title': img_tag.get('title', '').strip(),
            'width': img_tag.get('width', ''),
            'height': img_tag.get('height', '')
        }

        # Enhanced context extraction
        context_texts = []

        # Check figure caption
        figure_parent = img_tag.find_parent('figure')
        if figure_parent:
            figcaption = figure_parent.find('figcaption')
            if figcaption:
                context_texts.append(figcaption.get_text(strip=True))

        # Get surrounding text
        parent = img_tag.parent
        if parent:
            for sibling in parent.find_all(['p', 'span', 'div'], limit=3):
                text = sibling.get_text(strip=True)
                if text and len(text) > 10 and text not in context_texts:
                    context_texts.append(text[:100])

        img_info['semantic_context'] = context_texts

        # Nếu không có alt text, dùng context
        if context_texts and not img_info['alt']:
            img_info['alt'] = ' '.join(context_texts[:2])

        return img_info

    def download_image_with_enhanced_metadata(self, img_url, img_dir, filename_prefix, index, img_info):
        """Tải ảnh và chuẩn bị metadata chi tiết"""
        try:
            print(f"  Đang tải ảnh {index + 1}: {img_url}")
            
            # Rotate headers mỗi 5 requests
            if index % 5 == 0:
                self.update_session_headers()

            headers = {
                'User-Agent': self.session.headers['User-Agent'],
                'Referer': img_url,
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }

            response = self.session.get(
                img_url,
                timeout=self.config['SCRAPING']['timeout'],
                stream=True,
                headers=headers
            )

            response.raise_for_status()

            # Validate content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                print(f"    ⚠️ Không phải file ảnh: {content_type}")
                return None

            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.config['SCRAPING']['max_image_size']:
                print(f"    ⚠️ File quá lớn: {content_length} bytes")
                return None

            content = response.content
            if len(content) < self.config['SCRAPING']['min_image_size']:
                print(f"    ⚠️ File quá nhỏ: {len(content)} bytes")
                return None

            # Tạo hash để tránh trùng lặp
            img_hash = hashlib.md5(content).hexdigest()[:10]

            # Xác định extension
            ext_map = {
                'image/jpeg': '.jpg',
                'image/jpg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/webp': '.webp',
                'image/bmp': '.bmp',
                'image/svg+xml': '.svg',
                'image/x-icon': '.ico',
                'image/tiff': '.tiff'
            }

            ext = ext_map.get(content_type)
            if not ext:
                # Fallback: lấy từ URL
                parsed = urlparse(img_url)
                ext = os.path.splitext(parsed.path)[-1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico', '.tiff']:
                    ext = '.jpg'  # Default

            filename = f"{filename_prefix}_img_{index}_{img_hash}{ext}"
            img_path = img_dir / filename

            # Enhanced image metadata
            enhanced_metadata = {
                'url': img_url,
                'local_path': str(img_path),
                'alt': img_info.get('alt', ''),
                'title': img_info.get('title', ''),
                'semantic_context': img_info.get('semantic_context', []),
                'file_size': len(content),
                'content_type': content_type,
                'filename': filename,
                'hash': img_hash
            }

            # Process and save image
            try:
                if ext not in ['.svg', '.ico']:
                    # Validate image với PIL
                    img = Image.open(io.BytesIO(content))
                    width, height = img.size
                    
                    # Check minimum dimensions
                    min_width, min_height = self.config['SCRAPING']['min_image_dimensions']
                    if width < min_width or height < min_height:
                        print(f"    ⚠️ Ảnh quá nhỏ: {width}x{height}")
                        return None

                    enhanced_metadata['width'] = width
                    enhanced_metadata['height'] = height

                    # Verify image integrity
                    img.verify()

                    # Re-open for saving (verify() closes the image)
                    img = Image.open(io.BytesIO(content))
                    
                    # Convert RGBA to RGB if needed
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')

                    # Save with optimization
                    img.save(img_path, quality=85, optimize=True)
                else:
                    # Save SVG/ICO as-is
                    with open(img_path, 'wb') as f:
                        f.write(content)
                    enhanced_metadata['width'] = 0
                    enhanced_metadata['height'] = 0

                print(f"    ✅ Đã lưu: {filename}")
                return enhanced_metadata

            except Exception as e:
                # Fallback: save as binary
                try:
                    with open(img_path, 'wb') as f:
                        f.write(content)
                    enhanced_metadata['width'] = 0
                    enhanced_metadata['height'] = 0
                    print(f"    ✅ Đã lưu (raw): {filename}")
                    return enhanced_metadata
                except Exception as e2:
                    print(f"    ❌ Lỗi lưu file: {e2}")
                    return None

        except Exception as e:
            print(f"    ❌ Lỗi tải ảnh: {e}")
            return None

    def fetch_and_save(self, url):
        """Tải và lưu nội dung với enhanced processing"""
        max_retries = self.config['SCRAPING']['max_retries']
        
        for attempt in range(max_retries):
            try:
                print(f"\n🌐 Đang xử lý: {url} (Lần thử {attempt + 1}/{max_retries})")
                
                response = self.session.get(url, timeout=self.config['SCRAPING']['timeout'])
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith('text/html'):
                    print(f"⚠️ Không phải trang HTML: {content_type}")
                    return False

                # Detect encoding
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, "html.parser")

                # Tạo thư mục cho trang
                base_filename = self.clean_filename(url)
                page_dir = self.save_dir / base_filename

                if page_dir.exists():
                    print(f"🗑️ Xóa dữ liệu cũ: {page_dir.name}")
                    shutil.rmtree(page_dir, ignore_errors=True)

                page_dir.mkdir(exist_ok=True)
                print(f"📁 Tạo thư mục mới: {page_dir.name}")

                # Trích xuất enhanced metadata
                metadata = self.extract_enhanced_metadata(soup, url)
                print(f"📋 Title: {metadata['title']}")

                # Trích xuất semantic content
                clean_text = self.extract_semantic_content(soup)
                if not clean_text.strip():
                    print("⚠️ Không tìm thấy nội dung text")
                    clean_text = f"Nội dung từ {url}\nKhông thể trích xuất text từ trang này."

                metadata['word_count'] = len(clean_text.split())

                # Lưu text content
                text_file = page_dir / "content.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(clean_text)

                print(f"💾 Lưu text: {len(clean_text)} ký tự, {metadata['word_count']} từ")

                # Xử lý ảnh với enhanced metadata
                img_tags = soup.find_all("img")
                img_dir = page_dir / "images"
                img_dir.mkdir(exist_ok=True)

                print(f"🖼️ Tìm thấy {len(img_tags)} img tags")

                downloaded_images = []
                unique_urls = set()

                for idx, img_tag in enumerate(img_tags):
                    img_info = self.get_enhanced_image_info(img_tag, url)
                    if not img_info or img_info['url'] in unique_urls:
                        continue

                    unique_urls.add(img_info['url'])

                    img_metadata = self.download_image_with_enhanced_metadata(
                        img_info['url'],
                        img_dir,
                        base_filename,
                        idx,
                        img_info
                    )

                    if img_metadata:
                        downloaded_images.append(img_metadata)

                    # Delay between image downloads
                    time.sleep(0.2)

                metadata['image_count'] = len(downloaded_images)

                # Lưu metadata
                metadata_content = {
                    'metadata': metadata,
                    'images': downloaded_images
                }

                metadata_file = page_dir / "metadata.json"
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata_content, f, ensure_ascii=False, indent=2)

                print(f"💾 Lưu metadata: {metadata_file.name}")

                # Summary
                print(f"✅ Hoàn thành {url}")
                print(f"   📄 Text: {text_file.name}")
                print(f"   🖼️ Ảnh: {len(downloaded_images)} files")
                print(f"   📋 Metadata: {metadata_file.name}")

                self.success_count += 1
                return True

            except requests.exceptions.RequestException as e:
                print(f"❌ Lỗi kết nối {url} (Lần {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⏳ Chờ {wait_time} giây trước khi thử lại...")
                    time.sleep(wait_time)
                else:
                    self.error_count += 1
                    return False

            except Exception as e:
                print(f"❌ Lỗi khi xử lý {url}: {e}")
                print("Chi tiết lỗi:")
                traceback.print_exc()
                self.error_count += 1
                return False

        return False

    def print_summary(self):
        """In tóm tắt kết quả"""
        print(f"\n{'='*60}")
        print(f"📊 TÓM TẮT KẾT QUẢ")
        print(f"{'='*60}")
        print(f"✅ Thành công: {self.success_count}")
        print(f"❌ Lỗi: {self.error_count}")
        print(f"📁 Dữ liệu lưu tại: {self.save_dir.absolute()}")

        if self.save_dir.exists():
            subdirs = [d for d in self.save_dir.iterdir() if d.is_dir()]
            total_images = 0
            total_text_size = 0

            for subdir in subdirs:
                img_dir = subdir / "images"
                if img_dir.exists():
                    total_images += len(list(img_dir.glob("*")))

                text_file = subdir / "content.txt"
                if text_file.exists():
                    total_text_size += text_file.stat().st_size

            print(f"\n📂 THỐNG KÊ FILE:")
            print(f"   📂 Tổng số thư mục: {len(subdirs)}")
            print(f"   🖼️ Tổng số ảnh: {total_images}")
            print(f"   📝 Tổng dung lượng text: {total_text_size / 1024:.1f} KB")

        print(f"{'='*60}")

def main():
    """Hàm chính để thu thập dữ liệu"""
    try:
        print("🚀 Bắt đầu thu thập dữ liệu...")
        
        config = load_config_local()
        urls = config.get("URLS", [])

        if not urls:
            print("❌ Không tìm thấy URLs trong config_local!")
            print("Vui lòng kiểm tra file config_local.py và đảm bảo có danh sách URLS")
            return

        print(f"📋 Tìm thấy {len(urls)} URLs trong config")

        # Hỏi người dùng có muốn ghi đè không
        try:
            overwrite_choice = input("\n❓ Bạn có muốn ghi đè database cũ không? (y/N): ").lower().strip()
        except (EOFError, KeyboardInterrupt):
            overwrite_choice = 'n'

        overwrite = overwrite_choice in ['y', 'yes', 'có']

        if overwrite:
            print("🔄 Sẽ ghi đè database cũ")
        else:
            print("📂 Sẽ giữ lại dữ liệu cũ và chỉ cập nhật")

        # Hiển thị danh sách URLs
        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")

        # Khởi tạo scraper
        scraper = WebScraperLocal(overwrite=overwrite)

        print(f"\n🔄 Bắt đầu thu thập dữ liệu từ {len(urls)} URLs...")

        # Thu thập từng URL
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Xử lý URL: {url}")
            success = scraper.fetch_and_save(url)

            # Delay giữa các requests
            if i < len(urls):
                delay = scraper.config['SCRAPING']['delay_between_requests']
                print(f"⏳ Chờ {delay} giây trước khi xử lý URL tiếp theo...")
                time.sleep(delay)

        # In tóm tắt
        scraper.print_summary()

        if scraper.success_count > 0:
            print(f"\n🎉 Hoàn thành thu thập dữ liệu!")
            print(f"Bạn có thể chạy ứng dụng Streamlit để xem kết quả:")
            print(f"streamlit run app_local.py")
        else:
            print(f"\n😞 Không thu thập được dữ liệu nào!")

    except KeyboardInterrupt:
        print(f"\n⚠️ Người dùng dừng chương trình")
    except Exception as e:
        print(f"❌ Lỗi trong quá trình thực thi: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
