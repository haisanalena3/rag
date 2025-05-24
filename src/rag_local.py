import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from local_gemma import LocalGemmaClient
import logging

logger = logging.getLogger(__name__)

def advanced_preprocess_text(text):
    """Tiền xử lý văn bản nâng cao"""
    if not text:
        return ""
    
    text = text.lower()
    # Giữ lại các ký tự tiếng Việt
    text = re.sub(r'[^\w\s\-\.\àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_key_phrases(text, max_phrases=10):
    """Trích xuất cụm từ quan trọng"""
    if not text:
        return []
    
    words = text.split()
    if len(words) < 2:
        return words
    
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    
    phrase_counts = Counter(bigrams + trigrams)
    return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]

def calculate_semantic_similarity(query_words, doc_words):
    """Tính toán độ tương tự ngữ nghĩa"""
    if not query_words or not doc_words:
        return 0
    
    # Jaccard similarity
    intersection = len(query_words.intersection(doc_words))
    union = len(query_words.union(doc_words))
    jaccard = intersection / union if union > 0 else 0
    
    # Coverage similarity
    coverage = intersection / len(query_words) if len(query_words) > 0 else 0
    
    # Weighted combination
    return (jaccard * 0.6) + (coverage * 0.4)

def enhanced_search_documents(query, documents, threshold=0.2, top_k=3):
    """Tìm kiếm documents nâng cao với nhiều phương pháp"""
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
        
        # Phương pháp 1: TF-IDF với n-grams
        try:
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                stop_words=None,
                min_df=1,
                max_df=0.95
            )
            
            all_texts = list(valid_texts) + [processed_query]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            query_vector = tfidf_matrix[-1]
            doc_vectors = tfidf_matrix[:-1]
            
            tfidf_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}, using simple similarity")
            tfidf_similarities = np.zeros(len(valid_texts))
        
        # Phương pháp 2: Keyword matching với trọng số
        query_words = set(processed_query.split())
        keyword_scores = []
        
        for doc_text in valid_texts:
            doc_words = set(doc_text.split())
            score = calculate_semantic_similarity(query_words, doc_words)
            keyword_scores.append(score)
        
        keyword_scores = np.array(keyword_scores)
        
        # Phương pháp 3: Phrase matching
        query_phrases = extract_key_phrases(processed_query, 5)
        phrase_scores = []
        
        for doc_text in valid_texts:
            doc_phrases = extract_key_phrases(doc_text, 20)
            
            phrase_matches = 0
            for q_phrase in query_phrases:
                if any(q_phrase in d_phrase or d_phrase in q_phrase for d_phrase in doc_phrases):
                    phrase_matches += 1
            
            phrase_score = phrase_matches / len(query_phrases) if len(query_phrases) > 0 else 0
            phrase_scores.append(phrase_score)
        
        phrase_scores = np.array(phrase_scores)
        
        # Kết hợp các điểm số với trọng số
        final_scores = (
            tfidf_similarities * 0.5 +
            keyword_scores * 0.3 +
            phrase_scores * 0.2
        )
        
        max_similarity = np.max(final_scores) if len(final_scores) > 0 else 0
        
        # Lọc theo threshold
        results = []
        if max_similarity >= threshold:
            sorted_indices = np.argsort(final_scores)[::-1]
            for idx in sorted_indices[:top_k]:
                if final_scores[idx] >= threshold:
                    original_idx = valid_indices[idx]
                    results.append(documents[original_idx])
        
        return results, max_similarity
        
    except Exception as e:
        logger.error(f"Lỗi trong enhanced_search_documents: {e}")
        return [], 0

def search_documents_with_threshold(query, documents, threshold=0.2, top_k=3):
    """Wrapper function cho enhanced search"""
    return enhanced_search_documents(query, documents, threshold, top_k)

def adaptive_threshold_search(query, documents, base_threshold=0.2):
    """Tìm kiếm với threshold thích ứng"""
    if not query or not documents:
        return [], 0
    
    query_complexity = len(query.split())
    
    # Điều chỉnh threshold dựa trên độ phức tạp
    if query_complexity <= 3:
        threshold = base_threshold + 0.1
    elif query_complexity <= 7:
        threshold = base_threshold
    else:
        threshold = base_threshold - 0.1
    
    threshold = max(0.1, min(0.8, threshold))
    return enhanced_search_documents(query, documents, threshold)

def ask_local_model(question, context, config):
    """Hỏi model local với fallback"""
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
        
        logger.info("Sử dụng Gemma 3n local")
        
        response = local_client.generate_response(
            question, 
            context, 
            config["LOCAL_MODEL"]["max_tokens"],
            config["LOCAL_MODEL"]["temperature"],
            config["LOCAL_MODEL"]["top_p"],
            config["LOCAL_MODEL"]["timeout"]
        )
        
        return response
        
    except Exception as e:
        error_msg = f"Lỗi khi sử dụng model local: {e}"
        logger.error(error_msg)
        return error_msg

def load_documents(file_paths):
    """Load documents từ danh sách file paths"""
    documents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
        except Exception as e:
            logger.error(f"Lỗi đọc file {file_path}: {e}")
            documents.append("")
    return documents

def search_documents(query, documents, top_k=3):
    """Tìm kiếm documents liên quan nhất"""
    results, _ = enhanced_search_documents(query, documents, threshold=0.1, top_k=top_k)
    return results
