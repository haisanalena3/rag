import numpy as np
import sqlite3
import pickle
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import aiofiles

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector Database tối ưu cho multimodal RAG"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vector_config = config.get('VECTOR_DB', {})
        self.db_path = Path(self.vector_config.get('db_path', 'vector_database'))
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            self.vector_config.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        )
        
        # Vector storage
        self.vector_dimension = self.vector_config.get('vector_dimension', 384)
        self.similarity_threshold = self.vector_config.get('similarity_threshold', 0.5)
        
        # Initialize databases
        self.init_databases()
        self.load_existing_data()
    
    def init_databases(self):
        """Khởi tạo SQLite và FAISS databases"""
        self.metadata_db = sqlite3.connect(
            self.db_path / "metadata.db", 
            check_same_thread=False
        )
        self.init_metadata_tables()
        
        self.faiss_index = faiss.IndexFlatIP(self.vector_dimension)
        self.document_ids = []
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=None,
            min_df=1,
            max_df=0.95
        )
        self.tfidf_fitted = False
    
    def init_metadata_tables(self):
        """Khởi tạo bảng metadata"""
        cursor = self.metadata_db.cursor()
        
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
                embedding_id INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                url TEXT,
                local_path TEXT,
                alt_text TEXT,
                title TEXT,
                description TEXT,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                embedding_id INTEGER,
                created_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                query_type TEXT,
                results_count INTEGER,
                max_similarity REAL,
                search_time REAL,
                timestamp TEXT
            )
        """)
        
        self.metadata_db.commit()
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Tạo embedding cho text"""
        if not text or not text.strip():
            return np.zeros(self.vector_dimension)
        
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Lỗi tạo embedding: {e}")
            return np.zeros(self.vector_dimension)
    
    def add_document(self, doc_data: Dict) -> bool:
        """Thêm document vào vector database"""
        try:
            doc_id = doc_data['id']
            content_text = f"{doc_data.get('title', '')} {doc_data.get('description', '')} {doc_data.get('content', '')}"
            embedding = self.create_embedding(content_text)
            
            embedding_id = self.faiss_index.ntotal
            self.faiss_index.add(embedding.reshape(1, -1))
            self.document_ids.append(doc_id)
            
            cursor = self.metadata_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, url, title, description, content, content_type, language, author,
                 published_date, word_count, headings, keywords, metadata, embedding_id,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, doc_data.get('url', ''), doc_data.get('title', ''),
                doc_data.get('description', ''), doc_data.get('content', ''),
                doc_data.get('content_type', ''), doc_data.get('language', ''),
                doc_data.get('author', ''), doc_data.get('published_date', ''),
                doc_data.get('word_count', 0), json.dumps(doc_data.get('headings', [])),
                json.dumps(doc_data.get('keywords', [])), json.dumps(doc_data.get('metadata', {})),
                embedding_id, datetime.now().isoformat(), datetime.now().isoformat()
            ))
            
            self.metadata_db.commit()
            logger.info(f"Đã thêm document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi thêm document {doc_data.get('id', 'unknown')}: {e}")
            return False
    
    def add_image(self, img_data: Dict, document_id: str) -> bool:
        """Thêm image vào vector database"""
        try:
            img_id = img_data['id']
            img_text = f"{img_data.get('alt_text', '')} {img_data.get('title', '')} {img_data.get('description', '')}"
            embedding = self.create_embedding(img_text)
            
            embedding_id = self.faiss_index.ntotal
            self.faiss_index.add(embedding.reshape(1, -1))
            self.document_ids.append(f"img_{img_id}")
            
            cursor = self.metadata_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO images
                (id, document_id, url, local_path, alt_text, title, description,
                 width, height, file_size, embedding_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                img_id, document_id, img_data.get('url', ''), img_data.get('local_path', ''),
                img_data.get('alt_text', ''), img_data.get('title', ''), img_data.get('description', ''),
                img_data.get('width', 0), img_data.get('height', 0), img_data.get('file_size', 0),
                embedding_id, datetime.now().isoformat()
            ))
            
            self.metadata_db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Lỗi thêm image {img_data.get('id', 'unknown')}: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5, search_type: str = "hybrid") -> List[Dict]:
        """Tìm kiếm similar documents với hybrid search"""
        try:
            start_time = datetime.now()
            
            vector_results = []
            keyword_results = []
            
            if search_type in ["vector", "hybrid"]:
                vector_results = self._vector_search(query, top_k)
            if search_type in ["keyword", "hybrid"]:
                keyword_results = self._keyword_search(query, top_k)
            
            combined_results = self._combine_results(vector_results, keyword_results, top_k)
            
            search_time = (datetime.now() - start_time).total_seconds()
            self._log_search(query, search_type, len(combined_results), 
                           combined_results[0]['similarity'] if combined_results else 0, search_time)
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Lỗi tìm kiếm: {e}")
            return []
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Tìm kiếm vector similarity"""
        if self.faiss_index.ntotal == 0:
            return []
        
        query_embedding = self.create_embedding(query)
        similarities, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1), 
            min(top_k * 2, self.faiss_index.ntotal)
        )
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity < self.similarity_threshold:
                continue
                
            doc_id = self.document_ids[idx]
            doc_data = self._get_document_metadata(doc_id)
            
            if doc_data:
                results.append({
                    'id': doc_id,
                    'similarity': float(similarity),
                    'rank': i + 1,
                    'search_type': 'vector',
                    **doc_data
                })
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Tìm kiếm keyword với TF-IDF"""
        try:
            cursor = self.metadata_db.cursor()
            cursor.execute("SELECT id, title, description, content FROM documents")
            docs = cursor.fetchall()
            
            if not docs:
                return []
            
            corpus = []
            doc_ids = []
            for doc in docs:
                doc_id, title, description, content = doc
                text = f"{title} {description} {content}"
                corpus.append(text)
                doc_ids.append(doc_id)
            
            if not self.tfidf_fitted:
                self.tfidf_vectorizer.fit(corpus)
                self.tfidf_fitted = True
            
            doc_vectors = self.tfidf_vectorizer.transform(corpus)
            query_vector = self.tfidf_vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] < 0.3:  # Ngưỡng chặt chẽ hơn
                    continue
                
                doc_id = doc_ids[idx]
                doc_data = self._get_document_metadata(doc_id)
                
                if doc_data:
                    results.append({
                        'id': doc_id,
                        'similarity': float(similarities[idx]),
                        'rank': i + 1,
                        'search_type': 'keyword',
                        **doc_data
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Lỗi keyword search: {e}")
            return []
    
    def _combine_results(self, vector_results: List[Dict], keyword_results: List[Dict], top_k: int) -> List[Dict]:
        """Kết hợp kết quả vector và keyword search"""
        combined = {}
        
        for result in vector_results:
            doc_id = result['id']
            combined[doc_id] = result.copy()
            combined[doc_id]['combined_score'] = result['similarity'] * 0.6
        
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['combined_score'] += result['similarity'] * 0.4
                combined[doc_id]['search_type'] = 'hybrid'
            else:
                combined[doc_id] = result.copy()
                combined[doc_id]['combined_score'] = result['similarity'] * 0.4
        
        sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        return sorted_results[:top_k]
    
    def _get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Lấy metadata của document"""
        try:
            cursor = self.metadata_db.cursor()
            
            if doc_id.startswith('img_'):
                img_id = doc_id[4:]
                cursor.execute("SELECT * FROM images WHERE id = ?", (img_id,))
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
            else:
                cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
            
            return None
            
        except Exception as e:
            logger.error(f"Lỗi lấy metadata cho {doc_id}: {e}")
            return None
    
    def _log_search(self, query: str, query_type: str, results_count: int, max_similarity: float, search_time: float):
        """Log search history"""
        try:
            cursor = self.metadata_db.cursor()
            cursor.execute("""
                INSERT INTO search_history 
                (query, query_type, results_count, max_similarity, search_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (query, query_type, results_count, max_similarity, search_time, datetime.now().isoformat()))
            self.metadata_db.commit()
        except Exception as e:
            logger.error(f"Lỗi log search: {e}")
    
    def save_to_disk(self):
        """Lưu FAISS index và metadata vào disk"""
        try:
            faiss.write_index(self.faiss_index, str(self.db_path / "faiss_index.bin"))
            with open(self.db_path / "document_ids.pkl", 'wb') as f:
                pickle.dump(self.document_ids, f)
            logger.info("Đã lưu vector database")
        except Exception as e:
            logger.error(f"Lỗi lưu database: {e}")
    
    def load_existing_data(self):
        """Load dữ liệu hiện có từ disk"""
        try:
            faiss_file = self.db_path / "faiss_index.bin"
            ids_file = self.db_path / "document_ids.pkl"
            
            if faiss_file.exists() and ids_file.exists():
                self.faiss_index = faiss.read_index(str(faiss_file))
                with open(ids_file, 'rb') as f:
                    self.document_ids = pickle.load(f)
                logger.info(f"Đã load {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Lỗi load database: {e}")
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê database"""
        try:
            cursor = self.metadata_db.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM images")
            img_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM search_history")
            search_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(max_similarity) FROM search_history")
            avg_similarity = cursor.fetchone()[0] or 0
            
            return {
                'total_documents': doc_count,
                'total_images': img_count,
                'total_vectors': self.faiss_index.ntotal,
                'total_searches': search_count,
                'average_similarity': avg_similarity,
                'vector_dimension': self.vector_dimension,
                'similarity_threshold': self.similarity_threshold
            }
        except Exception as e:
            logger.error(f"Lỗi lấy thống kê: {e}")
            return {}