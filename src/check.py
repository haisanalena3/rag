# check_setup.py
import os
from pathlib import Path

def check_setup():
    print("🔍 Kiểm tra cấu hình hệ thống...")
    
    # Kiểm tra file config
    if os.path.exists("config.py"):
        print("✅ File config.py tồn tại")
        try:
            from config import load_config
            config = load_config()
            urls = config.get("URLS", [])
            print(f"📋 Tìm thấy {len(urls)} URLs trong config")
            for i, url in enumerate(urls, 1):
                print(f"  {i}. {url}")
        except Exception as e:
            print(f"❌ Lỗi đọc config: {e}")
    else:
        print("❌ File config.py không tồn tại")
    
    # Kiểm tra thư mục db
    db_dir = Path("db")
    if db_dir.exists():
        print("✅ Thư mục db tồn tại")
        subdirs = [d for d in db_dir.iterdir() if d.is_dir()]
        print(f"📁 Số thư mục con: {len(subdirs)}")
        
        for subdir in subdirs:
            print(f"\n📂 {subdir.name}:")
            metadata_file = subdir / "metadata.json"
            content_file = subdir / "content.txt"
            images_dir = subdir / "images"
            
            print(f"  metadata.json: {'✅' if metadata_file.exists() else '❌'}")
            print(f"  content.txt: {'✅' if content_file.exists() else '❌'}")
            print(f"  images/: {'✅' if images_dir.exists() else '❌'}")
            
            if images_dir.exists():
                img_count = len(list(images_dir.glob("*")))
                print(f"  Số ảnh: {img_count}")
    else:
        print("❌ Thư mục db không tồn tại")
        print("💡 Cần chạy collect.py để tạo database")

if __name__ == "__main__":
    check_setup()
