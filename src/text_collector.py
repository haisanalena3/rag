import hashlib
from datetime import datetime
import logging
from pathlib import Path
from rag_local import get_vector_database
from config_local import load_config_local
from rag_local import extract_key_phrases

logger = logging.getLogger(__name__)

class TextContentCollector:
    """Collector để lưu nội dung văn bản từ người dùng vào database"""
    
    def __init__(self, db_path="db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.config = load_config_local()
        
    def save_text_content(self, text_content: str) -> bool:
        """Lưu nội dung văn bản vào vector database"""
        try:
            vector_db = get_vector_database(self.config)
            
            # Tạo document data
            doc_id = hashlib.md5(text_content.encode()).hexdigest()
            timestamp = datetime.now().isoformat()
            doc_data = {
                'id': doc_id,
                'url': f'user_input_{doc_id}',
                'title': f"User Input {timestamp}",
                'description': text_content[:200] + "..." if len(text_content) > 200 else text_content,
                'content': text_content,
                'content_type': 'user_input',
                'language': 'vi',
                'author': 'User',
                'published_date': timestamp,
                'word_count': len(text_content.split()),
                'headings': [],
                'keywords': extract_key_phrases(text_content),
                'metadata': {
                    'source': 'user_input',
                    'created_at': timestamp
                }
            }
            
            # Thêm vào vector database
            success = vector_db.add_document(doc_data)
            if success:
                vector_db.save_to_disk()
                logger.info(f"Đã lưu document nhập từ người dùng {doc_id}")
                return True
            else:
                logger.error(f"Không thể lưu document {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi lưu nội dung văn bản: {e}")
            return False