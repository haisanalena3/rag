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
from config_local import load_config_local

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

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.success_count = 0
        self.error_count = 0

    def clean_filename(self, url):
        """Tạo tên file an toàn từ URL"""
        parsed = urlparse(url)
        filename = f"{parsed.netloc}_{parsed.path}".replace("/", "_").replace(".", "_")
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        return filename[:100]

    def extract_metadata(self, soup, url):
        """Trích xuất metadata từ HTML"""
        title = ""
        if soup.title:
            title = soup.title.string.strip()
        
        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            description = meta_desc.get("content", "")
        
        if not description:
            meta_desc = soup.find("meta", attrs={"property": "og:description"})
            if meta_desc:
                description = meta_desc.get("content", "")

        return {
            "title": title or f"Trang web từ {urlparse(url).netloc}",
            "description": description or "Không có mô tả",
            "url": url,
            "scraped_at": datetime.now().isoformat()
        }

    def clean_text(self, soup):
        """Trích xuất và làm sạch text từ HTML"""
        # Xóa script và style
        for script in soup(["script", "style"]):
            script.decompose()

        # Lấy text
        text = soup.get_text()
        
        # Làm sạch
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text

    def get_image_info(self, img_tag, base_url):
        """Lấy thông tin ảnh từ img tag"""
        src = img_tag.get('src') or img_tag.get('data-src')
        if not src:
            return None

        # Tạo URL đầy đủ
        img_url = urljoin(base_url, src)
        
        return {
            'url': img_url,
            'alt': img_tag.get('alt', ''),
            'title': img_tag.get('title', ''),
            'width': img_tag.get('width', ''),
            'height': img_tag.get('height', '')
        }

    def fetch_and_save(self, url):
        """Tải và lưu nội dung từ URL với ghi đè"""
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

            # Xóa thư mục cũ nếu tồn tại
            if page_dir.exists():
                print(f"🗑️ Xóa dữ liệu cũ: {page_dir.name}")
                shutil.rmtree(page_dir, ignore_errors=True)

            page_dir.mkdir(exist_ok=True)
            print(f"📁 Tạo thư mục mới: {page_dir.name}")

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

            # Đảm bảo metadata có đầy đủ thông tin
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

            # Ghi đè file metadata
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

    def download_image(self, img_url, img_dir, filename_prefix, index):
        """Tải và lưu ảnh với ghi đè"""
        try:
            print(f"   Đang tải ảnh {index + 1}: {img_url}")
            
            response = self.session.get(img_url, timeout=15, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                print(f"   ⚠️ Không phải file ảnh: {content_type}")
                return None

            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:
                print(f"   ⚠️ File quá lớn: {content_length} bytes")
                return None

            content = response.content
            if len(content) < 100:
                print(f"   ⚠️ File quá nhỏ: {len(content)} bytes")
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

            # Ghi đè ảnh nếu đã tồn tại
            try:
                if ext not in ['.svg', '.ico']:
                    img = Image.open(io.BytesIO(content))
                    width, height = img.size
                    if width < 50 or height < 50:
                        print(f"   ⚠️ Ảnh quá nhỏ: {width}x{height}")
                        return None

                    img.verify()
                    img = Image.open(io.BytesIO(content))
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(img_path, quality=85, optimize=True)
                else:
                    with open(img_path, 'wb') as f:
                        f.write(content)

                print(f"   ✅ Đã lưu: {filename}")
                return str(img_path)

            except Exception as e:
                try:
                    with open(img_path, 'wb') as f:
                        f.write(content)
                    print(f"   ✅ Đã lưu (raw): {filename}")
                    return str(img_path)
                except Exception as e2:
                    print(f"   ❌ Lỗi lưu file: {e2}")
                    return None

        except Exception as e:
            print(f"   ❌ Lỗi tải ảnh: {e}")
            return None

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

def clear_database(db_dir):
    """Xóa toàn bộ database"""
    db_path = Path(db_dir)
    if db_path.exists():
        print(f"🗑️ Đang xóa toàn bộ database: {db_path}")
        shutil.rmtree(db_path, ignore_errors=True)
        print(f"✅ Đã xóa database")
    else:
        print(f"ℹ️ Database không tồn tại: {db_path}")

def main():
    """Hàm chính với tùy chọn ghi đè"""
    try:
        print("🚀 Bắt đầu thu thập dữ liệu với Gemma Local...")
        
        config = load_config_local()
        urls = config.get("URLS", [])
        
        if not urls:
            print("❌ Không tìm thấy URLs trong config_local!")
            print("Vui lòng kiểm tra file config_local.py và đảm bảo có danh sách URLS")
            return

        print(f"📋 Tìm thấy {len(urls)} URLs trong config")

        # Hỏi người dùng có muốn ghi đè không
        overwrite_choice = input("\n❓ Bạn có muốn ghi đè database cũ không? (y/N): ").lower().strip()
        overwrite = overwrite_choice in ['y', 'yes', 'có']

        if overwrite:
            print("🔄 Sẽ ghi đè database cũ")
        else:
            print("📂 Sẽ giữ lại dữ liệu cũ và chỉ cập nhật")

        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")

        # Khởi tạo scraper với tùy chọn ghi đè
        scraper = WebScraperLocal(overwrite=overwrite)

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
            print(f"streamlit run app_local.py")
        else:
            print(f"\n😞 Không thu thập được dữ liệu nào!")
            print(f"Vui lòng kiểm tra:")
            print(f"- Kết nối internet")
            print(f"- URLs trong config_local.py có hợp lệ không")
            print(f"- Các website có thể truy cập được không")

    except KeyboardInterrupt:
        print(f"\n⚠️ Người dùng dừng chương trình")
    except Exception as e:
        print(f"❌ Lỗi trong quá trình thực thi: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
