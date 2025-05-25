import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from local_gemma import LocalGemmaClient
import logging
import time
import json

logger = logging.getLogger(__name__)

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

def enhanced_search_documents(query, documents, threshold=0.05, top_k=3):
    """Tìm kiếm documents nâng cao với threshold rất thấp"""
    if not documents or not query:
        return [], 0

    try:
        # Nếu documents là string, chuyển thành list
        if isinstance(documents, str):
            documents = [documents]

        processed_query = advanced_preprocess_text(query)
        
        # Xử lý documents - hỗ trợ cả string và dict
        processed_docs = []
        doc_objects = []
        
        for doc in documents:
            if isinstance(doc, dict):
                # Document object với metadata
                text_content = f"{doc.get('title', '')} {doc.get('description', '')} {doc.get('text_content', '')}"
                processed_docs.append(advanced_preprocess_text(text_content))
                doc_objects.append(doc)
            elif isinstance(doc, str):
                # Plain text document
                processed_docs.append(advanced_preprocess_text(doc))
                doc_objects.append({'text_content': doc, 'title': 'Document', 'description': '', 'metadata': {}})
        
        # Lọc documents hợp lệ
        valid_docs = [(i, doc) for i, doc in enumerate(processed_docs) if doc.strip()]
        if not valid_docs:
            return [], 0
        
        valid_indices, valid_texts = zip(*valid_docs)
        
        # TF-IDF với n-grams cải tiến
        try:
            vectorizer = TfidfVectorizer(
                max_features=5000,  # Giảm features để tăng tốc
                ngram_range=(1, 2),  # Giảm ngram range
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
        
        # Exact match bonus - quan trọng nhất
        exact_match_scores = []
        for doc_text in valid_texts:
            exact_matches = 0
            for word in query_words:
                if len(word) > 2 and word in doc_text:  # Giảm min length
                    exact_matches += 1
            exact_score = exact_matches / len(query_words) if len(query_words) > 0 else 0
            exact_match_scores.append(exact_score)
        
        exact_match_scores = np.array(exact_match_scores)
        
        # Kết hợp các điểm số với trọng số tối ưu - ưu tiên exact match
        final_scores = (
            tfidf_similarities * 0.3 +
            keyword_scores * 0.3 +
            exact_match_scores * 0.4  # Tăng trọng số exact match
        )
        
        max_similarity = np.max(final_scores) if len(final_scores) > 0 else 0
        
        # Sử dụng threshold rất thấp
        adjusted_threshold = max(0.01, threshold)
        
        # Lọc theo threshold
        results = []
        if max_similarity >= adjusted_threshold:
            sorted_indices = np.argsort(final_scores)[::-1]
            for idx in sorted_indices[:top_k]:
                if final_scores[idx] >= adjusted_threshold:
                    original_idx = valid_indices[idx]
                    results.append(doc_objects[original_idx])
        
        return results, max_similarity
        
    except Exception as e:
        logger.error(f"Lỗi trong enhanced_search_documents: {e}")
        return [], 0

def enhanced_search_with_metadata(query, documents, threshold=0.05, top_k=3):
    """Tìm kiếm nâng cao có xét đến metadata với threshold thấp"""
    if not documents or not query:
        return [], 0

    processed_query = advanced_preprocess_text(query)
    scored_docs = []

    for doc in documents:
        # Tính điểm cho text content
        text_content = f"{doc.get('title', '')}\n{doc.get('description', '')}\n{doc.get('text_content', '')}"
        processed_text = advanced_preprocess_text(text_content)
        
        # Tính điểm cho headings (trọng số cao hơn)
        headings_text = " ".join(doc.get('metadata', {}).get('headings', []))
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
        
        # Bonus cho title match - quan trọng
        title_score = 0
        if doc.get('title'):
            title_words = set(advanced_preprocess_text(doc['title']).split())
            query_words = set(processed_query.split())
            title_score = calculate_semantic_similarity(query_words, title_words) * 0.3
        
        # Exact word match bonus
        exact_bonus = 0
        query_words = processed_query.split()
        for word in query_words:
            if len(word) > 2 and word in processed_text:
                exact_bonus += 0.1
        
        # Tính điểm cuối với trọng số cải tiến
        final_score = (
            content_score * 0.4 +
            heading_score * 0.2 +
            title_score +
            exact_bonus
        )
        
        # Sử dụng threshold rất thấp
        if final_score >= 0.01:  # Threshold cực thấp
            scored_docs.append((doc, final_score))
    
    # Sắp xếp theo điểm
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    relevant_docs = [doc for doc, score in scored_docs[:top_k]]
    max_similarity = scored_docs[0][1] if scored_docs else 0
    
    return relevant_docs, max_similarity

def search_documents_with_threshold(query, documents, threshold=0.05, top_k=3):
    """Wrapper function cho enhanced search với threshold thấp"""
    return enhanced_search_documents(query, documents, threshold, top_k)

def adaptive_threshold_search(query, documents, base_threshold=0.05):
    """Tìm kiếm với threshold thích ứng cải tiến"""
    if not query or not documents:
        return [], 0

    query_complexity = len(query.split())
    query_lower = query.lower()
    
    # Điều chỉnh threshold dựa trên độ phức tạp và loại query
    if query_complexity <= 3:
        threshold = base_threshold
    elif query_complexity <= 7:
        threshold = base_threshold * 0.8
    else:
        threshold = base_threshold * 0.6
    
    # Điều chỉnh cho các loại query đặc biệt
    if any(keyword in query_lower for keyword in ['là gì', 'what is', 'define', 'định nghĩa']):
        threshold *= 0.7  # Dễ dàng hơn cho câu hỏi định nghĩa
    elif any(keyword in query_lower for keyword in ['cách', 'how to', 'làm thế nào']):
        threshold *= 0.7  # Dễ dàng hơn cho câu hỏi hướng dẫn
    
    threshold = max(0.01, min(0.5, threshold))
    
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
    """Tìm kiếm documents liên quan nhất với threshold rất thấp"""
    results, _ = enhanced_search_documents(query, documents, threshold=0.01, top_k=top_k)
    return results
