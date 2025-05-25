import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
import re
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sqlite3
from dataclasses import dataclass, asdict
import mimetypes

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MultimodalDocument:
    """Cấu trúc dữ liệu cho document multimodal"""
    id: str
    url: str
    title: str
    description: str
    content: str
    content_type: str
    language: str
    author: str
    published_date: str
    word_count: int
    headings: List[str]
    keywords: List[str]
    images: List[Dict]
    audio_files: List[Dict]
    video_files: List[Dict]
    metadata: Dict
    embedding: Optional[List[float]] = None
    created_at: str = ""
    updated_at: str = ""

@dataclass
class MediaItem:
    """Cấu trúc dữ liệu cho media item"""
    id: str
    url: str
    local_path: str
    media_type: str  # image, audio, video
    file_format: str
    size: int
    dimensions: Tuple[int, int] = None
    duration: float = None
    alt_text: str = ""
    title: str = ""
    description: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict = None

class EnhancedVectorDatabase:
    """Vector database tối ưu cho multimodal RAG"""
    
    def __init__(self, db_path: str = "vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # SQLite cho metadata
        self.metadata_db = sqlite3.connect(self.db_path / "metadata.db", check_same_thread=False)
        self.init_metadata_tables()
        
        # Vector storage
        self.text_vectors = {}
        self.image_vectors = {}
        self.audio_vectors = {}
        self.video_vectors = {}
        
        # TF-IDF vectorizer cho text
        self.text_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=None,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.load_existing_data()
    
    def init_metadata_tables(self):
        """Khởi tạo bảng metadata"""
        cursor = self.metadata_db.cursor()
        
        # Bảng documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                description TEXT,
                content TEXT,
                content_type TEXT,
                language TEXT,
                author TEXT,
                published_date TEXT,
                word_count INTEGER,
                headings TEXT,
                keywords TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Bảng media items
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS media_items (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                url TEXT,
                local_path TEXT,
                media_type TEXT,
                file_format TEXT,
                size INTEGER,
                dimensions TEXT,
                duration REAL,
                alt_text TEXT,
                title TEXT,
                description TEXT,
                metadata TEXT,
                created_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Bảng vector mappings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_mappings (
                id TEXT PRIMARY KEY,
                item_id TEXT,
                item_type TEXT,
                vector_file TEXT,
                embedding_model TEXT,
                created_at TEXT
            )
        """)
        
        self.metadata_db.commit()
    
    def save_document(self, doc: MultimodalDocument) -> bool:
        """Lưu document vào database"""
        try:
            cursor = self.metadata_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, url, title, description, content, content_type, language, author,
                 published_date, word_count, headings, keywords, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.url, doc.title, doc.description, doc.content,
                doc.content_type, doc.language, doc.author, doc.published_date,
                doc.word_count, json.dumps(doc.headings), json.dumps(doc.keywords),
                json.dumps(doc.metadata), doc.created_at, doc.updated_at
            ))
            
            # Lưu vector nếu có
            if doc.embedding:
                self.text_vectors[doc.id] = np.array(doc.embedding)
                self.save_vector_mapping(doc.id, "document", "text_embedding")
            
            self.metadata_db.commit()
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu document {doc.id}: {e}")
            return False
    
    def save_media_item(self, media: MediaItem, document_id: str) -> bool:
        """Lưu media item vào database"""
        try:
            cursor = self.metadata_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO media_items
                (id, document_id, url, local_path, media_type, file_format, size,
                 dimensions, duration, alt_text, title, description, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                media.id, document_id, media.url, media.local_path, media.media_type,
                media.file_format, media.size, json.dumps(media.dimensions),
                media.duration, media.alt_text, media.title, media.description,
                json.dumps(media.metadata or {}), datetime.now().isoformat()
            ))
            
            # Lưu vector theo loại media
            if media.embedding:
                if media.media_type == "image":
                    self.image_vectors[media.id] = np.array(media.embedding)
                elif media.media_type == "audio":
                    self.audio_vectors[media.id] = np.array(media.embedding)
                elif media.media_type == "video":
                    self.video_vectors[media.id] = np.array(media.embedding)
                
                self.save_vector_mapping(media.id, "media", f"{media.media_type}_embedding")
            
            self.metadata_db.commit()
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu media {media.id}: {e}")
            return False
    
    def save_vector_mapping(self, item_id: str, item_type: str, vector_type: str):
        """Lưu mapping vector"""
        cursor = self.metadata_db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO vector_mappings
            (id, item_id, item_type, vector_file, embedding_model, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            f"{item_id}_{vector_type}", item_id, item_type, vector_type,
            "tfidf_custom", datetime.now().isoformat()
        ))
    
    def load_existing_data(self):
        """Load dữ liệu vector hiện có"""
        try:
            # Load text vectors
            text_vector_file = self.db_path / "text_vectors.pkl"
            if text_vector_file.exists():
                with open(text_vector_file, 'rb') as f:
                    self.text_vectors = pickle.load(f)
            
            # Load image vectors
            image_vector_file = self.db_path / "image_vectors.pkl"
            if image_vector_file.exists():
                with open(image_vector_file, 'rb') as f:
                    self.image_vectors = pickle.load(f)
            
            logger.info(f"Loaded {len(self.text_vectors)} text vectors, {len(self.image_vectors)} image vectors")
        except Exception as e:
            logger.error(f"Lỗi load dữ liệu: {e}")
    
    def save_vectors_to_disk(self):
        """Lưu vectors vào disk"""
        try:
            with open(self.db_path / "text_vectors.pkl", 'wb') as f:
                pickle.dump(self.text_vectors, f)
            
            with open(self.db_path / "image_vectors.pkl", 'wb') as f:
                pickle.dump(self.image_vectors, f)
            
            with open(self.db_path / "audio_vectors.pkl", 'wb') as f:
                pickle.dump(self.audio_vectors, f)
            
            with open(self.db_path / "video_vectors.pkl", 'wb') as f:
                pickle.dump(self.video_vectors, f)
                
        except Exception as e:
            logger.error(f"Lỗi lưu vectors: {e}")
    
    def search_multimodal(self, query: str, modalities: List[str] = None, top_k: int = 5) -> List[Dict]:
        """Tìm kiếm multimodal với vector similarity"""
        if modalities is None:
            modalities = ["text", "image", "audio", "video"]
        
        results = []
        
        # Tìm kiếm text
        if "text" in modalities and self.text_vectors:
            text_results = self._search_text_vectors(query, top_k)
            results.extend(text_results)
        
        # Tìm kiếm image (dựa trên alt text và description)
        if "image" in modalities and self.image_vectors:
            image_results = self._search_image_vectors(query, top_k)
            results.extend(image_results)
        
        # Sắp xếp theo điểm similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _search_text_vectors(self, query: str, top_k: int) -> List[Dict]:
        """Tìm kiếm trong text vectors"""
        try:
            # Lấy tất cả documents
            cursor = self.metadata_db.cursor()
            cursor.execute("SELECT id, title, content FROM documents")
            docs = cursor.fetchall()
            
            if not docs:
                return []
            
            # Tạo corpus để fit vectorizer
            corpus = [doc[2] for doc in docs]  # content
            corpus.append(query)
            
            # Vectorize
            tfidf_matrix = self.text_vectorizer.fit_transform(corpus)
            query_vector = tfidf_matrix[-1]
            doc_vectors = tfidf_matrix[:-1]
            
            # Tính similarity
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            results = []
            for i, (doc_id, title, content) in enumerate(docs):
                if similarities[i] > 0.1:  # Threshold
                    results.append({
                        'id': doc_id,
                        'type': 'document',
                        'title': title,
                        'content': content[:200] + "...",
                        'similarity': float(similarities[i])
                    })
            
            return results
        except Exception as e:
            logger.error(f"Lỗi tìm kiếm text: {e}")
            return []
    
    def _search_image_vectors(self, query: str, top_k: int) -> List[Dict]:
        """Tìm kiếm trong image vectors dựa trên text description"""
        try:
            cursor = self.metadata_db.cursor()
            cursor.execute("""
                SELECT id, alt_text, title, description, local_path 
                FROM media_items 
                WHERE media_type = 'image'
            """)
            images = cursor.fetchall()
            
            if not images:
                return []
            
            # Tạo corpus từ alt text và description
            corpus = []
            for img in images:
                text_content = f"{img[1]} {img[2]} {img[3]}".strip()
                corpus.append(text_content if text_content else "image")
            
            corpus.append(query)
            
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vector = tfidf_matrix[-1]
            img_vectors = tfidf_matrix[:-1]
            
            # Tính similarity
            similarities = cosine_similarity(query_vector, img_vectors).flatten()
            
            results = []
            for i, (img_id, alt_text, title, description, local_path) in enumerate(images):
                if similarities[i] > 0.05:  # Threshold thấp hơn cho images
                    results.append({
                        'id': img_id,
                        'type': 'image',
                        'title': title or alt_text or "Image",
                        'description': description or alt_text,
                        'local_path': local_path,
                        'similarity': float(similarities[i])
                    })
            
            return results
        except Exception as e:
            logger.error(f"Lỗi tìm kiếm image: {e}")
            return []

class EnhancedMultimodalCollector:
    """Collector cải tiến cho multimodal RAG với vector database"""
    
    def __init__(self, db_path: str = "enhanced_multimodal_db", config: Dict = None):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Vector database
        self.vector_db = EnhancedVectorDatabase(self.db_path / "vectors")
        
        # Media storage
        self.media_dir = self.db_path / "media"
        self.media_dir.mkdir(exist_ok=True)
        
        # Config
        self.config = config or self._default_config()
        
        # Session với connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'images_downloaded': 0,
            'audio_files': 0,
            'video_files': 0,
            'errors': 0
        }
    
    def _default_config(self) -> Dict:
        """Cấu hình mặc định"""
        return {
            'timeout': 30,
            'max_retries': 3,
            'delay_between_requests': 1,
            'max_image_size': 20 * 1024 * 1024,  # 20MB
            'min_image_size': 1024,  # 1KB
            'supported_image_formats': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
            'supported_audio_formats': ['.mp3', '.wav', '.ogg', '.m4a'],
            'supported_video_formats': ['.mp4', '.avi', '.mov', '.webm'],
            'max_concurrent_downloads': 5,
            'enable_image_analysis': True,
            'enable_audio_analysis': False,
            'enable_video_analysis': False
        }
    
    async def collect_from_urls(self, urls: List[str], batch_size: int = 5) -> Dict:
        """Thu thập dữ liệu từ danh sách URLs với async processing"""
        logger.info(f"Bắt đầu thu thập từ {len(urls)} URLs")
        
        results = {
            'success': [],
            'failed': [],
            'total_documents': 0,
            'total_media': 0
        }
        
        # Xử lý theo batch
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            logger.info(f"Xử lý batch {i//batch_size + 1}: {len(batch)} URLs")
            
            # Async processing cho batch
            async with aiohttp.ClientSession() as session:
                tasks = [self._process_url_async(session, url) for url in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Xử lý kết quả batch
            for url, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Lỗi xử lý {url}: {result}")
                    results['failed'].append({'url': url, 'error': str(result)})
                    self.stats['errors'] += 1
                elif result:
                    results['success'].append(result)
                    results['total_documents'] += 1
                    results['total_media'] += len(result.get('media_items', []))
                else:
                    results['failed'].append({'url': url, 'error': 'Unknown error'})
            
            # Delay giữa các batch
            if i + batch_size < len(urls):
                await asyncio.sleep(self.config['delay_between_requests'])
        
        # Lưu vectors vào disk
        self.vector_db.save_vectors_to_disk()
        
        logger.info(f"Hoàn thành thu thập: {len(results['success'])} thành công, {len(results['failed'])} thất bại")
        return results
    
    async def _process_url_async(self, session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
        """Xử lý một URL với async"""
        try:
            # Fetch HTML content
            async with session.get(url, timeout=self.config['timeout']) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                
                html_content = await response.text()
            
            # Parse và extract
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Tạo document
            doc = self._create_document_from_soup(soup, url)
            
            # Extract media
            media_items = await self._extract_media_async(session, soup, url, doc.id)
            
            # Tạo embeddings
            doc.embedding = self._create_text_embedding(doc.content)
            
            # Lưu vào database
            if self.vector_db.save_document(doc):
                for media in media_items:
                    self.vector_db.save_media_item(media, doc.id)
                
                self.stats['documents_processed'] += 1
                return {
                    'document': asdict(doc),
                    'media_items': [asdict(media) for media in media_items]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Lỗi xử lý URL {url}: {e}")
            raise e
    
    def _create_document_from_soup(self, soup: BeautifulSoup, url: str) -> MultimodalDocument:
        """Tạo document từ BeautifulSoup object"""
        # Extract metadata
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        content = self._extract_clean_text(soup)
        headings = self._extract_headings(soup)
        keywords = self._extract_keywords(soup, content)
        
        # Tạo ID unique
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Detect content type
        content_type = self._detect_content_type(url, soup)
        
        # Extract other metadata
        author = self._extract_author(soup)
        published_date = self._extract_published_date(soup)
        language = self._extract_language(soup)
        
        return MultimodalDocument(
            id=doc_id,
            url=url,
            title=title,
            description=description,
            content=content,
            content_type=content_type,
            language=language,
            author=author,
            published_date=published_date,
            word_count=len(content.split()),
            headings=headings,
            keywords=keywords,
            images=[],
            audio_files=[],
            video_files=[],
            metadata={
                'scraped_at': datetime.now().isoformat(),
                'scraper_version': 'enhanced_v1.0'
            },
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    async def _extract_media_async(self, session: aiohttp.ClientSession, soup: BeautifulSoup, 
                                 base_url: str, doc_id: str) -> List[MediaItem]:
        """Extract và download media files async"""
        media_items = []
        
        # Extract images
        img_tags = soup.find_all('img')
        image_tasks = []
        
        for img_tag in img_tags[:10]:  # Giới hạn 10 ảnh đầu tiên
            img_info = self._extract_image_info(img_tag, base_url)
            if img_info:
                task = self._download_image_async(session, img_info, doc_id)
                image_tasks.append(task)
        
        # Download images concurrently
        if image_tasks:
            downloaded_images = await asyncio.gather(*image_tasks, return_exceptions=True)
            for result in downloaded_images:
                if isinstance(result, MediaItem):
                    media_items.append(result)
                    self.stats['images_downloaded'] += 1
        
        # Extract audio (nếu enable)
        if self.config['enable_audio_analysis']:
            audio_elements = soup.find_all(['audio', 'source'])
            for audio_elem in audio_elements:
                audio_info = self._extract_audio_info(audio_elem, base_url)
                if audio_info:
                    # Download audio file
                    audio_media = await self._download_audio_async(session, audio_info, doc_id)
                    if audio_media:
                        media_items.append(audio_media)
                        self.stats['audio_files'] += 1
        
        # Extract video (nếu enable)
        if self.config['enable_video_analysis']:
            video_elements = soup.find_all(['video', 'iframe'])
            for video_elem in video_elements:
                video_info = self._extract_video_info(video_elem, base_url)
                if video_info:
                    # Download video file hoặc extract thumbnail
                    video_media = await self._process_video_async(session, video_info, doc_id)
                    if video_media:
                        media_items.append(video_media)
                        self.stats['video_files'] += 1
        
        return media_items
    
    async def _download_image_async(self, session: aiohttp.ClientSession, 
                                  img_info: Dict, doc_id: str) -> Optional[MediaItem]:
        """Download image file async"""
        try:
            async with session.get(img_info['url'], timeout=self.config['timeout']) as response:
                if response.status != 200:
                    return None
                
                content = await response.read()
                
                # Validate image
                if len(content) < self.config['min_image_size'] or len(content) > self.config['max_image_size']:
                    return None
                
                # Tạo filename và path
                img_id = hashlib.md5(f"{doc_id}_{img_info['url']}".encode()).hexdigest()
                file_ext = self._get_file_extension(img_info['url'], response.headers.get('content-type', ''))
                filename = f"{img_id}{file_ext}"
                local_path = self.media_dir / "images" / filename
                local_path.parent.mkdir(exist_ok=True)
                
                # Lưu file
                with open(local_path, 'wb') as f:
                    f.write(content)
                
                # Analyze image
                dimensions = None
                if self.config['enable_image_analysis']:
                    dimensions = self._analyze_image(local_path)
                
                # Tạo embedding cho image (dựa trên alt text)
                embedding = None
                if img_info.get('alt') or img_info.get('title'):
                    text_content = f"{img_info.get('alt', '')} {img_info.get('title', '')}".strip()
                    embedding = self._create_text_embedding(text_content)
                
                return MediaItem(
                    id=img_id,
                    url=img_info['url'],
                    local_path=str(local_path),
                    media_type='image',
                    file_format=file_ext,
                    size=len(content),
                    dimensions=dimensions,
                    alt_text=img_info.get('alt', ''),
                    title=img_info.get('title', ''),
                    description=img_info.get('description', ''),
                    embedding=embedding,
                    metadata={
                        'downloaded_at': datetime.now().isoformat(),
                        'content_type': response.headers.get('content-type', '')
                    }
                )
                
        except Exception as e:
            logger.error(f"Lỗi download image {img_info['url']}: {e}")
            return None
    
    def _create_text_embedding(self, text: str) -> List[float]:
        """Tạo embedding cho text sử dụng TF-IDF"""
        if not text or not text.strip():
            return [0.0] * 100  # Vector rỗng
        
        try:
            # Sử dụng TF-IDF đơn giản
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            vector = vectorizer.fit_transform([text.lower()])
            return vector.toarray()[0].tolist()
        except:
            return [0.0] * 100
    
    def _analyze_image(self, image_path: Path) -> Tuple[int, int]:
        """Phân tích thông tin ảnh"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except:
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title từ HTML"""
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
                    return title.strip()
        
        return "Untitled Document"
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract description từ HTML"""
        desc_sources = [
            soup.find('meta', attrs={'name': 'description'}),
            soup.find('meta', attrs={'property': 'og:description'}),
            soup.find('meta', attrs={'name': 'twitter:description'})
        ]
        
        for source in desc_sources:
            if source and source.get('content'):
                return source.get('content').strip()
        
        return ""
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract và làm sạch text content"""
        # Loại bỏ các thẻ không cần thiết
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
        
        # Tìm main content
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
            text = soup.get_text(separator="\n", strip=True)
        
        # Làm sạch text
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 5:
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract headings từ HTML"""
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        return [h.get_text(strip=True) for h in headings if h.get_text(strip=True)]
    
    def _extract_keywords(self, soup: BeautifulSoup, content: str) -> List[str]:
        """Extract keywords từ meta tags và content"""
        keywords = []
        
        # Từ meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and keywords_tag.get('content'):
            meta_keywords = [k.strip() for k in keywords_tag.get('content').split(',')]
            keywords.extend(meta_keywords)
        
        # Từ content (simple keyword extraction)
        if content:
            words = re.findall(r'\b\w{4,}\b', content.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Lấy top 10 từ xuất hiện nhiều nhất
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords.extend([word for word, freq in top_words if freq > 2])
        
        return list(set(keywords))[:20]  # Giới hạn 20 keywords
    
    def _detect_content_type(self, url: str, soup: BeautifulSoup) -> str:
        """Detect loại content"""
        url_lower = url.lower()
        
        if any(keyword in url_lower for keyword in ['blog', 'post', 'article']):
            return 'blog'
        elif any(keyword in url_lower for keyword in ['news', 'tin-tuc']):
            return 'news'
        elif any(keyword in url_lower for keyword in ['tutorial', 'guide', 'huong-dan']):
            return 'tutorial'
        elif any(keyword in url_lower for keyword in ['doc', 'documentation']):
            return 'documentation'
        else:
            return 'general'
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author từ HTML"""
        author_sources = [
            soup.find('meta', attrs={'name': 'author'}),
            soup.find('meta', attrs={'property': 'article:author'}),
            soup.find('meta', attrs={'name': 'twitter:creator'})
        ]
        
        for source in author_sources:
            if source and source.get('content'):
                return source.get('content').strip()
        
        return ""
    
    def _extract_published_date(self, soup: BeautifulSoup) -> str:
        """Extract published date từ HTML"""
        date_sources = [
            soup.find('meta', attrs={'property': 'article:published_time'}),
            soup.find('meta', attrs={'name': 'publish_date'}),
            soup.find('time', attrs={'datetime': True})
        ]
        
        for source in date_sources:
            if source:
                date_val = source.get('content') or source.get('datetime')
                if date_val:
                    return date_val.strip()
        
        return ""
    
    def _extract_language(self, soup: BeautifulSoup) -> str:
        """Extract language từ HTML"""
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            return html_tag.get('lang')
        return "unknown"
    
    def _extract_image_info(self, img_tag, base_url: str) -> Optional[Dict]:
        """Extract thông tin image từ img tag"""
        img_url = (img_tag.get('src') or 
                  img_tag.get('data-src') or 
                  img_tag.get('data-original'))
        
        if not img_url:
            return None
        
        img_url = urljoin(base_url, img_url)
        
        # Validate URL
        if not self._is_valid_image_url(img_url):
            return None
        
        return {
            'url': img_url,
            'alt': img_tag.get('alt', '').strip(),
            'title': img_tag.get('title', '').strip(),
            'description': ''
        }
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Kiểm tra URL ảnh hợp lệ"""
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
            
            # Kiểm tra extension
            path = parsed.path.lower()
            return any(path.endswith(ext) for ext in self.config['supported_image_formats'])
        except:
            return False
    
    def _get_file_extension(self, url: str, content_type: str) -> str:
        """Lấy file extension từ URL hoặc content-type"""
        # Từ URL
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext in self.config['supported_image_formats']:
            return ext
        
        # Từ content-type
        if content_type:
            ext = mimetypes.guess_extension(content_type.split(';')[0])
            if ext and ext in self.config['supported_image_formats']:
                return ext
        
        return '.jpg'  # Default
    
    def search(self, query: str, modalities: List[str] = None, top_k: int = 5) -> List[Dict]:
        """Tìm kiếm multimodal"""
        return self.vector_db.search_multimodal(query, modalities, top_k)
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê thu thập"""
        return {
            **self.stats,
            'total_documents': len(self.vector_db.text_vectors),
            'total_images': len(self.vector_db.image_vectors),
            'database_size': self._calculate_db_size()
        }
    
    def _calculate_db_size(self) -> Dict:
        """Tính toán kích thước database"""
        try:
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(self.db_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            return {
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count
            }
        except:
            return {'total_size_mb': 0, 'file_count': 0}

# Async helper functions
async def _extract_audio_info(self, audio_elem, base_url: str) -> Optional[Dict]:
    """Extract audio information"""
    # Implementation for audio extraction
    pass

async def _download_audio_async(self, session: aiohttp.ClientSession, 
                              audio_info: Dict, doc_id: str) -> Optional[MediaItem]:
    """Download audio file async"""
    # Implementation for audio download
    pass

async def _extract_video_info(self, video_elem, base_url: str) -> Optional[Dict]:
    """Extract video information"""
    # Implementation for video extraction
    pass

async def _process_video_async(self, session: aiohttp.ClientSession,
                             video_info: Dict, doc_id: str) -> Optional[MediaItem]:
    """Process video async"""
    # Implementation for video processing
    pass

# Main execution
async def main():
    """Hàm main để test collector"""
    collector = EnhancedMultimodalCollector()
    
    # Test URLs
    test_urls = [
        "https://thangnotes.dev/2023/09/30/bai-1-tim-hieu-ve-saga-design-pattern/",
        "https://thangnotes.dev/2023/08/12/phan-2-cai-dat-zookeeper-etcd-va-cach-trien-khai-leadership-election/"
    ]
    
    # Thu thập dữ liệu
    results = await collector.collect_from_urls(test_urls)
    
    print("=== KẾT QUẢ THU THẬP ===")
    print(f"Thành công: {len(results['success'])}")
    print(f"Thất bại: {len(results['failed'])}")
    print(f"Tổng documents: {results['total_documents']}")
    print(f"Tổng media: {results['total_media']}")
    
    # Test tìm kiếm
    print("\n=== TEST TÌM KIẾM ===")
    search_results = collector.search("design pattern", ["text", "image"], 3)
    for result in search_results:
        print(f"- {result['title']}: {result['similarity']:.3f}")
    
    # Thống kê
    print("\n=== THỐNG KÊ ===")
    stats = collector.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
