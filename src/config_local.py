def load_config_local():
    return {
        "URLS": [
            "https://thangnotes.dev/2023/09/30/bai-1-tim-hieu-ve-saga-design-pattern/",
            "https://thangnotes.dev/2023/08/12/phan-2-cai-dat-zookeeper-etcd-va-cach-trien-khai-leadership-election/",
            "https://thangnotes.dev/2023/08/06/phan-1-tim-hieu-ve-cac-he-thong-phan-tan-voi-zookeeper-etcd/",
            "https://thangnotes.dev/2023/08/04/tai-file-pdf-bi-chan-tai-xuong-tren-drive-va-get-link-cac-video-tai-ve/",
            "https://thangnotes.dev/2023/05/25/1-1-thuc-hanh-cai-dat-database-oracle-19c-tren-ec2-aws/",
        ],

        # Cấu hình cho Local Gemma
        "USE_LOCAL_MODEL": True,
        "LOCAL_MODEL": {
            "base_url": "http://localhost:11434",
            "model": "gemma3:1b-it-qat",  # Có thể thay bằng gemma3:1b hoặc gemma3:8b
            "max_tokens": 2000,
            "temperature": 0.3,
            "top_p": 0.4,
            "timeout": 180
        },

        # Vector Database Configuration - FIXED
        "VECTOR_DB": {
            "enabled": False,  # Tạm tắt để dùng traditional search
            "db_path": "vector_database",
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "vector_dimension": 384,
            "similarity_threshold": 0.1,  # Giảm threshold
            "max_results": 10,
            "index_type": "faiss",
            "distance_metric": "cosine"
        },

        # Cấu hình database - FIXED
        "DB_THRESHOLD": 0.05,  # Giảm threshold xuống rất thấp
        "MAX_SEARCH_RESULTS": 5,
        "MAX_IMAGES_PER_RESPONSE": 8,

        # Cấu hình scraping nâng cao
        "SCRAPING": {
            "timeout": 20,
            "max_retries": 3,
            "delay_between_requests": 1.5,
            "max_image_size": 15 * 1024 * 1024,  # 15MB
            "min_image_size": 512,  # 512 bytes
            "min_image_dimensions": (32, 32),
            "user_agents": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ],
            "headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive"
            }
        },

        # Advanced Search Configuration - FIXED
        "SEARCH": {
            "enable_semantic_search": True,
            "enable_keyword_search": True,
            "enable_fuzzy_search": True,
            "search_weights": {
                "title": 0.3,
                "content": 0.4,
                "headings": 0.2,
                "metadata": 0.1
            },
            "reranking_enabled": True,
            "query_expansion": True
        }
    }
