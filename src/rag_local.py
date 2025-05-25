import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from local_gemma import LocalGemmaClient
import logging
import time
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Import vector database với fallback
try:
    import faiss
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logger.warning("FAISS không có sẵn, sử dụng traditional search")

class VectorDatabase:
    """Vector Database local cho RAG"""
    
    def __init__(self, config, db_path="vector_db"):
        self.config = config
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Vector storage
        self.vector_dimension = 384
        self.similarity_threshold = 0.5
        self.documents = {}
        self.document_ids = []
        
        # Initialize components
        if VECTOR_DB_AVAILABLE:
            self.init_vector_components()
        else:
            self.init_fallback_components()
        
        self.load_existing_data()
    
    def init_vector_components(self):
        """Khởi tạo vector components"""
        try:
            # FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.vector_dimension)
            
            # SQLite for metadata
            self.metadata_db = sqlite3.connect(
                self.db_path / "metadata.db", 
                check_same_thread=False
            )
            self.init_metadata_tables()
            
            # TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.vector_dimension,
                ngram_range=(1, 3),
                stop_words=None,
                min_df=1,
                max_df=0.95
            )
            self.tfidf_fitted = False
            
            logger.info("Vector database components initialized")
        except Exception as e:
            logger.error(f"Error initializing vector components: {e}")
            self.init_fallback_components()
    
    def init_fallback_components(self):
        """Khởi tạo fallback components"""
        self.faiss_index = None
        self.metadata_db = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None
        )
        self.tfidf_fitted = False
        logger.info("Fallback components initialized")
    
    def init_metadata_tables(self):
        """Khởi tạo bảng metadata"""
        if not self.metadata_db:
            return
            
        cursor = self.metadata_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT,
                title TEXT,
                description TEXT,
                content TEXT,
                metadata TEXT,
                embedding_id INTEGER,
                created_at TEXT
            )
        """)
        self.metadata_db.commit()
    
    def create_embedding(self, text):
        """Tạo embedding cho text với xử lý tiếng Việt"""
        if not text or not text.strip():
            return np.zeros(self.vector_dimension)
        
        try:
            # Tiền xử lý text
            processed_text = self.preprocess_vietnamese_text(text)
            
            # Sử dụng TF-IDF để tạo embedding
            if not self.tfidf_fitted:
                corpus = list(self.documents.values()) + [processed_text]
                self.tfidf_vectorizer.fit(corpus)
                self.tfidf_fitted = True
            
            vector = self.tfidf_vectorizer.transform([processed_text])
            dense_vector = vector.toarray()[0]
            
            # Pad hoặc truncate để match dimension
            if len(dense_vector) < self.vector_dimension:
                padded = np.zeros(self.vector_dimension)
                padded[:len(dense_vector)] = dense_vector
                return padded.astype(np.float32)
            else:
                return dense_vector[:self.vector_dimension].astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return np.zeros(self.vector_dimension)
    
    def preprocess_vietnamese_text(self, text):
        """Tiền xử lý text tiếng Việt"""
        text = text.lower()
        text = re.sub(r'[^\w\s\-\.\àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        text = re.sub(r'\b\d+\b', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def add_document(self, doc_data):
        """Thêm document vào vector database"""
        try:
            doc_id = doc_data['id']
            content_text = f"{doc_data.get('title', '')} {doc_data.get('description', '')} {doc_data.get('content', '')}"
            
            # Store document
            self.documents[doc_id] = content_text
            
            # Create embedding
            embedding = self.create_embedding(content_text)
            
            # Add to FAISS if available
            if self.faiss_index is not None:
                embedding_id = self.faiss_index.ntotal
                self.faiss_index.add(embedding.reshape(1, -1))
                self.document_ids.append(doc_id)
            
            # Save to SQLite if available
            if self.metadata_db:
                cursor = self.metadata_db.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, url, title, description, content, metadata, embedding_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, doc_data.get('url', ''), doc_data.get('title', ''),
                    doc_data.get('description', ''), doc_data.get('content', ''),
                    json.dumps(doc_data.get('metadata', {})),
                    self.faiss_index.ntotal - 1 if self.faiss_index else -1,
                    time.strftime('%Y-%m-%d %H:%M:%S')
                ))
                self.metadata_db.commit()
            
            logger.info(f"Added document {doc_id} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_data.get('id', 'unknown')}: {e}")
            return False
    
    def search_similar(self, query, top_k=5):
        """Tìm kiếm similar documents với hybrid approach"""
        try:
            if not self.documents:
                return []
            
            processed_query = self.preprocess_vietnamese_text(query)
            query_embedding = self.create_embedding(processed_query)
            
            # Vector search với FAISS
            if self.faiss_index is not None and self.faiss_index.ntotal > 0:
                similarities, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1), 
                    min(top_k * 2, self.faiss_index.ntotal)
                )
                
                results = []
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if similarity >= 0.1 and idx < len(self.document_ids):
                        doc_id = self.document_ids[idx]
                        doc_data = self._get_document_metadata(doc_id)
                        if doc_data:
                            results.append({
                                'id': doc_id,
                                'similarity': float(similarity),
                                'rank': i + 1,
                                **doc_data
                            })
                
                return results[:top_k]
            
            # Fallback to TF-IDF similarity
            else:
                return self._fallback_search(processed_query, top_k)
                
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query, top_k):
        """Fallback search using TF-IDF"""
        try:
            if not self.documents:
                return []
            
            doc_ids = list(self.documents.keys())
            doc_texts = list(self.documents.values())
            
            # Create TF-IDF matrix
            all_texts = doc_texts + [query]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            query_vector = tfidf_matrix[-1]
            doc_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0.05:
                    doc_id = doc_ids[idx]
                    doc_data = self._get_document_metadata(doc_id)
                    if doc_data:
                        results.append({
                            'id': doc_id,
                            'similarity': float(similarities[idx]),
                            'rank': i + 1,
                            **doc_data
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
    
    def _get_document_metadata(self, doc_id):
        """Lấy metadata của document"""
        try:
            if self.metadata_db:
                cursor = self.metadata_db.cursor()
                cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
            
            # Fallback: return basic info
            return {
                'id': doc_id,
                'title': f"Document {doc_id}",
                'content': self.documents.get(doc_id, ''),
                'url': '',
                'description': ''
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata for {doc_id}: {e}")
            return None
    
    def load_existing_data(self):
        """Load dữ liệu hiện có từ disk"""
        try:
            # Load FAISS index
            faiss_file = self.db_path / "faiss_index.bin"
            if faiss_file.exists() and VECTOR_DB_AVAILABLE:
                self.faiss_index = faiss.read_index(str(faiss_file))
            
            # Load document IDs
            ids_file = self.db_path / "document_ids.pkl"
            if ids_file.exists():
                import pickle
                with open(ids_file, 'rb') as f:
                    self.document_ids = pickle.load(f)
            
            # Load documents
            docs_file = self.db_path / "documents.pkl"
            if docs_file.exists():
                import pickle
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
            
            logger.info(f"Loaded {len(self.documents)} documents from disk")
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
    
    def save_to_disk(self):
        """Lưu vector database vào disk"""
        try:
            import pickle
            
            if self.faiss_index:
                faiss.write_index(self.faiss_index, str(self.db_path / "faiss_index.bin"))
            
            with open(self.db_path / "document_ids.pkl", 'wb') as f:
                pickle.dump(self.document_ids, f)
            
            with open(self.db_path / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info("Vector database saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")

# Global vector database instance
_vector_db = None

def get_vector_database(config):
    """Lấy instance vector database (singleton pattern)"""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDatabase(config)
    return _vector_db

def advanced_preprocess_text(text):
    """Tiền xử lý văn bản nâng cao với xử lý tiếng Việt tốt hơn"""
    if not text:
        return ""
    
    text = text.lower()
    # Giữ lại các ký tự tiếng Việt và loại bỏ ký tự đặc biệt
    text = re.sub(r'[^\w\s\-\.\àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    # Loại bỏ số đơn lẻ và ký tự đặc biệt
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_key_phrases(text, max_phrases=15):
    """Trích xuất cụm từ quan trọng với cải tiến"""
    if not text:
        return []
    
    words = text.split()
    if len(words) < 2:
        return words
    
    # Tạo n-grams
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    
    # Lọc các phrase có ý nghĩa
    meaningful_phrases = []
    for phrase in bigrams + trigrams:
        # Loại bỏ phrase chỉ chứa stop words hoặc quá ngắn
        if len(phrase) > 4 and not re.match(r'^(và|của|cho|với|từ|tại|trong|trên|dưới|về|là|có|được|đã|sẽ|này|đó|khi|nếu|như|để|theo|sau|trước|giữa|ngoài|bên|qua|lên|xuống)\s', phrase):
            meaningful_phrases.append(phrase)
    
    phrase_counts = Counter(meaningful_phrases)
    return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]

def calculate_semantic_similarity(query_words, doc_words):
    """Tính toán độ tương tự ngữ nghĩa với cải tiến"""
    if not query_words or not doc_words:
        return 0
    
    # Jaccard similarity
    intersection = len(query_words.intersection(doc_words))
    union = len(query_words.union(doc_words))
    jaccard = intersection / union if union > 0 else 0
    
    # Coverage similarity (bao phủ query)
    coverage = intersection / len(query_words) if len(query_words) > 0 else 0
    
    # Precision similarity (độ chính xác)
    precision = intersection / len(doc_words) if len(doc_words) > 0 else 0
    
    # Weighted combination với trọng số cải tiến
    return (jaccard * 0.4) + (coverage * 0.4) + (precision * 0.2)

def enhanced_search_with_vector_db(query, config, threshold=0.2, top_k=3):
    """Tìm kiếm nâng cao sử dụng vector database"""
    try:
        vector_db = get_vector_database(config)
        
        # Tìm kiếm trong vector database
        vector_results = vector_db.search_similar(query, top_k * 2)
        
        if not vector_results:
            return [], 0
        
        # Chuyển đổi kết quả về format cũ để tương thích
        relevant_docs = []
        max_similarity = 0
        
        for result in vector_results:
            if result['similarity'] >= threshold:
                # Reconstruct document format
                doc = {
                    'id': result['id'],
                    'url': result.get('url', ''),
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'text_content': result.get('content', ''),
                    'metadata': {},
                    'images': [],  # Will be populated separately
                    'search_weight': result['similarity']
                }
                
                # Parse metadata if available
                try:
                    if result.get('metadata'):
                        if isinstance(result['metadata'], str):
                            doc['metadata'] = json.loads(result['metadata'])
                        else:
                            doc['metadata'] = result['metadata']
                except:
                    doc['metadata'] = {}
                
                relevant_docs.append(doc)
                max_similarity = max(max_similarity, result['similarity'])
        
        return relevant_docs[:top_k], max_similarity
        
    except Exception as e:
        logger.error(f"Lỗi tìm kiếm vector database: {e}")
        return [], 0

def enhanced_search_documents(query, documents, threshold=0.2, top_k=3):
    """Tìm kiếm documents nâng cao - fallback cho vector DB"""
    if not documents or not query:
        return [], 0

    try:
        processed_query = advanced_preprocess_text(query)
        processed_docs = [advanced_preprocess_text(doc) for doc in documents]
        
        # Lọc documents hợp lệ
        valid_docs = [(i, doc) for i, doc in enumerate(processed_docs) if doc.strip()]
        if not valid_docs:
            return [], 0
        
        valid_indices, valid_texts = zip(*valid_docs)
        
        # TF-IDF với n-grams cải tiến
        try:
            vectorizer = TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 3),
                stop_words=None,
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                use_idf=True
            )
            
            all_texts = list(valid_texts) + [processed_query]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1]
            doc_vectors = tfidf_matrix[:-1]
            
            tfidf_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}, using simple similarity")
            tfidf_similarities = np.zeros(len(valid_texts))
        
        # Keyword matching với trọng số cải tiến
        query_words = set(processed_query.split())
        keyword_scores = []
        
        for doc_text in valid_texts:
            doc_words = set(doc_text.split())
            score = calculate_semantic_similarity(query_words, doc_words)
            keyword_scores.append(score)
        
        keyword_scores = np.array(keyword_scores)
        
        # Phrase matching cải tiến
        query_phrases = extract_key_phrases(processed_query, 8)
        phrase_scores = []
        
        for doc_text in valid_texts:
            doc_phrases = extract_key_phrases(doc_text, 25)
            total_phrase_score = 0
            
            for q_phrase in query_phrases:
                best_match_score = 0
                for d_phrase in doc_phrases:
                    if q_phrase in d_phrase or d_phrase in q_phrase:
                        match_score = min(len(q_phrase), len(d_phrase)) / max(len(q_phrase), len(d_phrase))
                        best_match_score = max(best_match_score, match_score)
                total_phrase_score += best_match_score
            
            phrase_score = total_phrase_score / len(query_phrases) if len(query_phrases) > 0 else 0
            phrase_scores.append(phrase_score)
        
        phrase_scores = np.array(phrase_scores)
        
        # Exact match bonus
        exact_match_scores = []
        for doc_text in valid_texts:
            exact_matches = 0
            for word in query_words:
                if len(word) > 3 and word in doc_text:
                    exact_matches += 1
            exact_score = exact_matches / len(query_words) if len(query_words) > 0 else 0
            exact_match_scores.append(exact_score)
        
        exact_match_scores = np.array(exact_match_scores)
        
        # Kết hợp các điểm số với trọng số tối ưu
        final_scores = (
            tfidf_similarities * 0.4 +
            keyword_scores * 0.25 +
            phrase_scores * 0.25 +
            exact_match_scores * 0.1
        )
        
        max_similarity = np.max(final_scores) if len(final_scores) > 0 else 0
        
        # Điều chỉnh threshold dựa trên độ phức tạp query
        query_complexity = len(processed_query.split())
        adjusted_threshold = threshold
        
        if query_complexity <= 3:
            adjusted_threshold = threshold + 0.05
        elif query_complexity >= 10:
            adjusted_threshold = threshold - 0.05
        
        # Lọc theo threshold
        results = []
        if max_similarity >= adjusted_threshold:
            sorted_indices = np.argsort(final_scores)[::-1]
            for idx in sorted_indices[:top_k]:
                if final_scores[idx] >= adjusted_threshold:
                    original_idx = valid_indices[idx]
                    results.append(documents[original_idx])
        
        return results, max_similarity
        
    except Exception as e:
        logger.error(f"Lỗi trong enhanced_search_documents: {e}")
        return [], 0

def enhanced_search_with_metadata(query, documents, threshold=0.2, top_k=3):
    """Tìm kiếm nâng cao có xét đến metadata - sử dụng vector DB nếu có"""
    # Try vector database first
    from config_local import load_config_local
    config = load_config_local()
    
    if config.get('VECTOR_DB', {}).get('enabled', False):
        return enhanced_search_with_vector_db(query, config, threshold, top_k)
    
    # Fallback to traditional search
    if not documents or not query:
        return [], 0

    processed_query = advanced_preprocess_text(query)
    scored_docs = []

    for doc in documents:
        # Tính điểm cho text content
        text_content = f"{doc['title']}\n{doc['description']}\n{doc['text_content']}"
        processed_text = advanced_preprocess_text(text_content)
        
        # Tính điểm cho headings (trọng số cao hơn)
        headings_text = " ".join(doc['metadata'].get('headings', []))
        processed_headings = advanced_preprocess_text(headings_text)
        
        # Tính điểm tổng hợp
        content_score = calculate_semantic_similarity(
            set(processed_query.split()),
            set(processed_text.split())
        )
        
        heading_score = calculate_semantic_similarity(
            set(processed_query.split()),
            set(processed_headings.split())
        )
        
        # Bonus cho content type phù hợp
        content_type_bonus = 0
        query_lower = processed_query.lower()
        content_type = doc['metadata'].get('content_type', '')
        
        if 'tutorial' in query_lower or 'hướng dẫn' in query_lower:
            if content_type == 'tutorial':
                content_type_bonus = 0.15
        elif 'news' in query_lower or 'tin tức' in query_lower:
            if content_type == 'news':
                content_type_bonus = 0.15
        elif 'blog' in query_lower:
            if content_type == 'blog':
                content_type_bonus = 0.1
        
        # Bonus cho title match
        title_score = 0
        if doc['title']:
            title_words = set(advanced_preprocess_text(doc['title']).split())
            query_words = set(processed_query.split())
            title_score = calculate_semantic_similarity(query_words, title_words) * 0.2
        
        # Tính điểm cuối
        final_score = (
            content_score * 0.5 +
            heading_score * 0.25 +
            title_score +
            content_type_bonus
        )
        
        if final_score >= threshold:
            scored_docs.append((doc, final_score))
    
    # Sắp xếp theo điểm
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    relevant_docs = [doc for doc, score in scored_docs[:top_k]]
    max_similarity = scored_docs[0][1] if scored_docs else 0
    
    return relevant_docs, max_similarity

def search_documents_with_threshold(query, documents, threshold=0.2, top_k=3):
    """Wrapper function cho enhanced search"""
    return enhanced_search_documents(query, documents, threshold, top_k)

def adaptive_threshold_search(query, documents, base_threshold=0.2):
    """Tìm kiếm với threshold thích ứng cải tiến"""
    if not query or not documents:
        return [], 0

    query_complexity = len(query.split())
    query_lower = query.lower()
    
    # Điều chỉnh threshold dựa trên độ phức tạp và loại query
    if query_complexity <= 3:
        threshold = base_threshold + 0.1
    elif query_complexity <= 7:
        threshold = base_threshold
    else:
        threshold = base_threshold - 0.1
    
    # Điều chỉnh cho các loại query đặc biệt
    if any(keyword in query_lower for keyword in ['là gì', 'what is', 'define', 'định nghĩa']):
        threshold -= 0.05  # Dễ dàng hơn cho câu hỏi định nghĩa
    elif any(keyword in query_lower for keyword in ['cách', 'how to', 'làm thế nào']):
        threshold -= 0.05  # Dễ dàng hơn cho câu hỏi hướng dẫn
    elif any(keyword in query_lower for keyword in ['so sánh', 'compare', 'khác nhau']):
        threshold += 0.05  # Khó hơn cho câu hỏi so sánh
    
    threshold = max(0.1, min(0.8, threshold))
    
    return enhanced_search_documents(query, documents, threshold)

def ask_local_model(question, context, config):
    """Hỏi model local với fallback và retry logic"""
    if not config.get("USE_LOCAL_MODEL", False):
        return "Model local không được kích hoạt trong cấu hình."

    try:
        local_client = LocalGemmaClient(
            base_url=config["LOCAL_MODEL"]["base_url"],
            model=config["LOCAL_MODEL"]["model"]
        )
        
        # Kiểm tra kết nối
        if not local_client.check_connection():
            return "Không thể kết nối với Ollama server. Vui lòng kiểm tra xem Ollama có đang chạy không."
        
        # Kiểm tra model
        if not local_client.check_model_exists(config["LOCAL_MODEL"]["model"]):
            available_models = local_client.get_available_models()
            return f"Model {config['LOCAL_MODEL']['model']} không tồn tại. Models có sẵn: {', '.join(available_models)}"
        
        logger.info("Sử dụng Gemma local")
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = local_client.generate_response(
                    question,
                    context,
                    config["LOCAL_MODEL"]["max_tokens"],
                    config["LOCAL_MODEL"]["temperature"],
                    config["LOCAL_MODEL"]["top_p"],
                    config["LOCAL_MODEL"]["timeout"]
                )
                
                if response and len(response.strip()) > 10:
                    return response
                else:
                    logger.warning(f"Response quá ngắn hoặc rỗng, thử lại lần {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Lỗi lần thử {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise e
        
        return "Không thể tạo câu trả lời sau nhiều lần thử."
        
    except Exception as e:
        error_msg = f"Lỗi khi sử dụng model local: {e}"
        logger.error(error_msg)
        return error_msg

def load_documents(file_paths):
    """Load documents từ danh sách file paths với xử lý lỗi"""
    documents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    documents.append(content)
                else:
                    logger.warning(f"File rỗng: {file_path}")
                    documents.append("")
        except Exception as e:
            logger.error(f"Lỗi đọc file {file_path}: {e}")
            documents.append("")
    return documents

def search_documents(query, documents, top_k=3):
    """Tìm kiếm documents liên quan nhất với threshold thấp"""
    results, _ = enhanced_search_documents(query, documents, threshold=0.1, top_k=top_k)
    return results
