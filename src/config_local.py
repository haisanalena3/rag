def load_config_local():
    return {
        "URLS": [
            "https://thangnotes.dev/2023/09/30/bai-1-tim-hieu-ve-saga-design-pattern/",
            "https://thangnotes.dev/2023/08/12/phan-2-cai-dat-zookeeper-etcd-va-cach-trien-khai-leadership-election/",
            "https://thangnotes.dev/2023/08/06/phan-1-tim-hieu-ve-cac-he-thong-phan-tan-voi-zookeeper-etcd/",
            "https://thangnotes.dev/2023/08/04/tai-file-pdf-bi-chan-tai-xuong-tren-drive-va-get-link-cac-video-tai-ve/",
            "https://thangnotes.dev/2023/05/25/1-1-thuc-hanh-cai-dat-database-oracle-19c-tren-ec2-aws/",
            # Thêm các URL khác bạn muốn thu thập
        ],
        # Cấu hình cho Local Gemma
        "USE_LOCAL_MODEL": True,
        "LOCAL_MODEL": {
            "base_url": "https://api-ai.thangnotes.dev/",
            "model": "gemma3:1b-it-qat",  # Có thể thay bằng gemma3:1b hoặc gemma3:8b
            "max_tokens": 1000,
            "temperature": 0.4,
            "top_p": 0.4,
            "timeout": 90
        },
        # Backup Gemini API (optional)
        "GEMINI_API_KEY": "your_backup_gemini_key_here",
        "MODEL_NAME": "gemini-pro",
        # Cấu hình database
        "DB_THRESHOLD": 0.3,
        "MAX_SEARCH_RESULTS": 3,
        "MAX_IMAGES_PER_RESPONSE": 5
    }
