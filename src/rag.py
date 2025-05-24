import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import re

def preprocess_text(text):
    """Tiền xử lý văn bản"""
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt, giữ lại chữ cái, số và khoảng trắng
    text = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    return text

def load_documents(file_paths):
    """Load documents từ danh sách file paths"""
    documents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
        except Exception as e:
            print(f"Lỗi đọc file {file_path}: {e}")
            documents.append("")
    return documents

def search_documents(query, documents, top_k=3):
    """Tìm kiếm documents liên quan nhất"""
    if not documents:
        return []
    
    try:
        # Tiền xử lý query và documents
        processed_query = preprocess_text(query)
        processed_docs = [preprocess_text(doc) for doc in documents]
        
        # Loại bỏ documents rỗng
        valid_docs = [(i, doc) for i, doc in enumerate(processed_docs) if doc.strip()]
        
        if not valid_docs:
            return []
        
        valid_indices, valid_texts = zip(*valid_docs)
        
        # Tạo TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # Không sử dụng stop words cho tiếng Việt
            ngram_range=(1, 2)
        )
        
        # Fit vectorizer với documents và query
        all_texts = list(valid_texts) + [processed_query]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Tính cosine similarity
        query_vector = tfidf_matrix[-1]  # Vector cuối cùng là query
        doc_vectors = tfidf_matrix[:-1]  # Các vector còn lại là documents
        
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Sắp xếp theo độ tương tự
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Lấy top_k documents có similarity > 0
        results = []
        for idx in sorted_indices[:top_k]:
            if similarities[idx] > 0:
                original_idx = valid_indices[idx]
                results.append(documents[original_idx])
        
        return results
        
    except Exception as e:
        print(f"Lỗi trong search_documents: {e}")
        return []

def search_documents_with_threshold(query, documents, threshold=0.3, top_k=3):
    """Tìm kiếm documents với ngưỡng độ liên quan"""
    if not documents:
        return [], 0
    
    try:
        # Tiền xử lý
        processed_query = preprocess_text(query)
        processed_docs = [preprocess_text(doc) for doc in documents]
        
        # Loại bỏ documents rỗng
        valid_docs = [(i, doc) for i, doc in enumerate(processed_docs) if doc.strip()]
        
        if not valid_docs:
            return [], 0
        
        valid_indices, valid_texts = zip(*valid_docs)
        
        # Tạo TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        all_texts = list(valid_texts) + [processed_query]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Tính cosine similarity
        query_vector = tfidf_matrix[-1]
        doc_vectors = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Tính điểm trung bình
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        
        # Lọc theo threshold
        results = []
        if max_similarity >= threshold:
            sorted_indices = np.argsort(similarities)[::-1]
            for idx in sorted_indices[:top_k]:
                if similarities[idx] >= threshold:
                    original_idx = valid_indices[idx]
                    results.append(documents[original_idx])
        
        return results, max_similarity
        
    except Exception as e:
        print(f"Lỗi trong search_documents_with_threshold: {e}")
        return [], 0

def ask_gemini(question, context, api_key, model_name="gemini-pro"):
    """Hỏi Gemini với context"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        if context.strip():
            prompt = f"""
Dựa trên thông tin sau đây, hãy trả lời câu hỏi một cách chính xác và chi tiết:

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Hãy trả lời bằng tiếng Việt và dựa trên thông tin đã cung cấp. Nếu thông tin không đủ để trả lời, hãy nói rõ điều đó.
"""
        else:
            prompt = f"""
Câu hỏi: {question}

Hãy trả lời câu hỏi này bằng tiếng Việt một cách chi tiết và chính xác.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Xin lỗi, tôi không thể trả lời câu hỏi này. Lỗi: {e}"
