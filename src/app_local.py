import os
import json
import streamlit as st
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import traceback
import numpy as np
import re
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import time
from datetime import datetime
import shutil

from config_local import load_config_local
from rag_local import (
    load_documents, search_documents, ask_local_model,
    search_documents_with_threshold, adaptive_threshold_search
)
from local_gemma import LocalGemmaClient
from collect_local import WebScraperLocal

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMultimodalRAGLocal:
    def __init__(self, db_dir):
        self.db_dir = Path(db_dir)
        self.documents = []
        self.has_data = False
        self.query_history = []
        self.load_multimodal_data()

    def load_multimodal_data(self):
        """Load dữ liệu đa phương tiện từ database"""
        self.documents = []
        logger.info(f"Kiểm tra thư mục db: {self.db_dir.absolute()}")
        
        if not self.db_dir.exists():
            logger.warning(f"Thư mục db không tồn tại: {self.db_dir}")
            self.has_data = False
            return

        site_dirs = [d for d in self.db_dir.iterdir() if d.is_dir()]
        logger.info(f"Tìm thấy {len(site_dirs)} thư mục con trong db")

        for site_dir in site_dirs:
            logger.info(f"Kiểm tra thư mục: {site_dir.name}")
            metadata_file = site_dir / "metadata.json"
            
            if not metadata_file.exists():
                logger.warning(f"Thiếu metadata.json trong {site_dir.name}")
                continue

            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                content_file = site_dir / "content.txt"
                text_content = ""
                if content_file.exists():
                    with open(content_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()

                doc = {
                    'site_name': site_dir.name,
                    'url': data['metadata']['url'],
                    'title': data['metadata']['title'],
                    'description': data['metadata']['description'],
                    'text_content': text_content,
                    'images': data.get('images', []),
                    'image_dir': site_dir / "images",
                    'metadata': data['metadata']
                }

                self.documents.append(doc)
                logger.info(f"Thêm document: {doc['title']}")

            except Exception as e:
                logger.error(f"Lỗi load dữ liệu từ {site_dir}: {e}")

        self.has_data = len(self.documents) > 0
        logger.info(f"Tổng cộng: {len(self.documents)} documents")

    def analyze_query_intent(self, query):
        """Phân tích ý định của câu hỏi"""
        query_lower = query.lower()
        
        intent_keywords = {
            'visual': ['ảnh', 'hình', 'màu', 'nhìn', 'thấy', 'hiển thị', 'minh họa', 'hình ảnh'],
            'descriptive': ['mô tả', 'giải thích', 'là gì', 'như thế nào', 'tại sao', 'định nghĩa'],
            'comparative': ['so sánh', 'khác nhau', 'giống', 'tương tự', 'hơn', 'khác biệt'],
            'instructional': ['cách', 'làm', 'thực hiện', 'bước', 'hướng dẫn', 'phương pháp'],
            'factual': ['khi nào', 'ở đâu', 'ai', 'bao nhiêu', 'số lượng', 'thống kê']
        }

        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score

        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'general'
        return primary_intent, intent_scores

    def enhanced_search_multimodal(self, query, threshold=0.2, top_k=3):
        """Tìm kiếm đa phương tiện nâng cao"""
        if not self.has_data:
            return [], 0

        primary_intent, intent_scores = self.analyze_query_intent(query)

        text_corpus = []
        doc_weights = []

        for doc in self.documents:
            combined_text = f"{doc['title']}\n{doc['description']}\n{doc['text_content']}"
            
            image_descriptions = []
            for img in doc['images']:
                if img.get('alt'):
                    image_descriptions.append(img['alt'])
                if img.get('title'):
                    image_descriptions.append(img['title'])

            # Điều chỉnh trọng số dựa trên intent
            if primary_intent == 'visual' and image_descriptions:
                combined_text += "\n" + "\n".join(image_descriptions) * 2
                doc_weights.append(1.5)
            elif primary_intent == 'descriptive':
                combined_text = doc['description'] + "\n" + combined_text
                doc_weights.append(1.2)
            else:
                combined_text += "\n" + "\n".join(image_descriptions)
                doc_weights.append(1.0)

            text_corpus.append(combined_text)

        try:
            relevant_texts, max_similarity = adaptive_threshold_search(
                query, text_corpus, threshold
            )

            relevant_docs = []
            for relevant_text in relevant_texts:
                for i, text in enumerate(text_corpus):
                    if text == relevant_text:
                        doc = self.documents[i].copy()
                        doc['search_weight'] = doc_weights[i]
                        relevant_docs.append(doc)
                        break

            relevant_docs.sort(key=lambda x: x.get('search_weight', 1.0), reverse=True)

            # Lưu lịch sử
            self.query_history.append({
                'query': query,
                'intent': primary_intent,
                'similarity': max_similarity,
                'results_count': len(relevant_docs)
            })

            return relevant_docs[:top_k], max_similarity

        except Exception as e:
            logger.error(f"Lỗi enhanced search: {e}")
            return [], 0

    def get_relevant_images_for_context(self, relevant_docs, query, max_images=5):
        """Lấy ảnh liên quan để đưa vào context"""
        relevant_images = []
        
        for doc in relevant_docs:
            for img_info in doc['images']:
                relevance_score = 0
                alt_text = img_info.get('alt', '').lower()
                title_text = img_info.get('title', '').lower()
                query_lower = query.lower()

                # Tính điểm dựa trên từ khóa
                for word in query_lower.split():
                    if word in alt_text or word in title_text:
                        relevance_score += 1

                # Bonus điểm cho ảnh có mô tả chi tiết
                if len(alt_text) > 20 or len(title_text) > 20:
                    relevance_score += 0.5

                if relevance_score > 0 or (not alt_text and not title_text):
                    img_path = Path(img_info.get('local_path', ''))
                    if img_path.exists():
                        relevant_images.append({
                            'path': img_path,
                            'alt': img_info.get('alt', ''),
                            'title': img_info.get('title', ''),
                            'url': img_info.get('url', ''),
                            'source_doc': doc,
                            'relevance_score': relevance_score
                        })

        relevant_images.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_images[:max_images]

    def get_search_statistics(self):
        """Thống kê hiệu suất tìm kiếm"""
        if not self.query_history:
            return {}

        total_queries = len(self.query_history)
        successful_queries = sum(1 for q in self.query_history if q['results_count'] > 0)
        avg_similarity = np.mean([q['similarity'] for q in self.query_history])

        intent_distribution = {}
        for q in self.query_history:
            intent = q['intent']
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1

        return {
            'total_queries': total_queries,
            'success_rate': successful_queries / total_queries * 100,
            'average_similarity': avg_similarity,
            'intent_distribution': intent_distribution
        }

    def get_database_statistics(self):
        """Thống kê database"""
        if not self.has_data:
            return {}
        
        total_images = 0
        total_text_length = 0
        sites_info = []
        
        for doc in self.documents:
            total_images += len(doc['images'])
            total_text_length += len(doc['text_content'])
            
            # Tính kích thước thư mục
            site_dir = self.db_dir / doc['site_name']
            site_size = 0
            if site_dir.exists():
                for file_path in site_dir.rglob('*'):
                    if file_path.is_file():
                        site_size += file_path.stat().st_size
            
            sites_info.append({
                'name': doc['title'],
                'url': doc['url'],
                'images': len(doc['images']),
                'text_length': len(doc['text_content']),
                'size_bytes': site_size,
                'scraped_at': doc['metadata'].get('scraped_at', 'N/A')
            })
        
        return {
            'total_documents': len(self.documents),
            'total_images': total_images,
            'total_text_length': total_text_length,
            'sites_info': sites_info
        }

def check_ollama_connection(base_url):
    """Kiểm tra kết nối tới Ollama server"""
    try:
        # Đảm bảo URL đúng format
        if isinstance(base_url, dict):
            logger.error(f"base_url không đúng format: {base_url}")
            return False, "URL không đúng format", "base_url phải là string"
        
        clean_url = str(base_url).rstrip('/')
        if not clean_url.startswith(('http://', 'https://')):
            clean_url = f"https://{clean_url}"
        
        logger.info(f"Kiểm tra kết nối tới: {clean_url}")
        
        # Thử endpoint health check
        response = requests.get(clean_url, timeout=5)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            return True, "Kết nối thành công", response.text.strip()
        else:
            return False, f"Server phản hồi không đúng: {response.status_code}", response.text
            
    except requests.exceptions.ConnectionError:
        return False, "Không thể kết nối tới server", "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout khi kết nối", "Request timeout"
    except Exception as e:
        logger.error(f"Lỗi kiểm tra kết nối: {e}")
        return False, f"Lỗi không xác định: {str(e)}", str(e)

def get_ollama_models(base_url):
    """Lấy danh sách models từ Ollama"""
    try:
        clean_url = str(base_url).rstrip('/')
        if not clean_url.startswith(('http://', 'https://')):
            clean_url = f"https://{clean_url}"
            
        models_url = f"{clean_url}/api/tags"
        logger.info(f"Gọi API tags: {models_url}")
        
        response = requests.get(models_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get('models', [])
        else:
            logger.warning(f"API tags trả về status: {response.status_code}")
            return False, []
            
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách models: {e}")
        return False, []

def get_running_models(base_url):
    """Lấy danh sách models đang chạy"""
    try:
        clean_url = str(base_url).rstrip('/')
        if not clean_url.startswith(('http://', 'https://')):
            clean_url = f"https://{clean_url}"
            
        ps_url = f"{clean_url}/api/ps"
        response = requests.get(ps_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            # Làm sạch dữ liệu models
            cleaned_models = []
            for model in models:
                if isinstance(model, dict):
                    cleaned_model = {
                        'name': str(model.get('name', 'Unknown')),
                        'size': model.get('size'),
                        'expires_at': model.get('expires_at')
                    }
                    cleaned_models.append(cleaned_model)
            
            return True, cleaned_models
        else:
            return False, []
            
    except Exception as e:
        logger.error(f"Lỗi khi lấy running models: {e}")
        return False, []

def get_model_info(base_url, model_name):
    """Lấy thông tin chi tiết của model"""
    try:
        clean_url = str(base_url).rstrip('/')
        if not clean_url.startswith(('http://', 'https://')):
            clean_url = f"https://{clean_url}"
            
        show_url = f"{clean_url}/api/show"
        payload = {"name": model_name}
        response = requests.post(show_url, json=payload, timeout=15)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {}
            
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin model {model_name}: {e}")
        return False, {}

def format_model_size(size_bytes):
    """Format kích thước model"""
    if size_bytes is None:
        return "N/A"
    
    try:
        if isinstance(size_bytes, str):
            size_bytes = int(size_bytes)
        elif not isinstance(size_bytes, (int, float)):
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"
    
    if size_bytes <= 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def format_duration(nanoseconds):
    """Format thời gian từ nanoseconds"""
    if nanoseconds is None:
        return "N/A"
    
    try:
        if isinstance(nanoseconds, str):
            nanoseconds = int(nanoseconds)
        elif not isinstance(nanoseconds, (int, float)):
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"
    
    if nanoseconds <= 0:
        return "0s"
    
    seconds = nanoseconds / 1_000_000_000
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def should_use_general_knowledge(relevant_docs, similarity, question):
    """Kiểm tra xem có nên sử dụng kiến thức chung không"""
    MIN_SIMILARITY = 0.2
    
    if not relevant_docs:
        return True
    
    if similarity < MIN_SIMILARITY:
        return True
    
    total_content_length = 0
    for doc in relevant_docs:
        content = doc.get('text_content', '') + doc.get('description', '')
        total_content_length += len(content.strip())
    
    if total_content_length < 100:
        return True
    
    question_words = set(question.lower().split())
    content_words = set()
    
    for doc in relevant_docs:
        content = doc.get('text_content', '') + doc.get('description', '') + doc.get('title', '')
        content_words.update(content.lower().split())
    
    matching_words = question_words.intersection(content_words)
    match_ratio = len(matching_words) / len(question_words) if question_words else 0
    
    if match_ratio < 0.1:
        return True
    
    return False

def create_intelligent_multimodal_context(text_context, relevant_images, question):
    """Tạo context đa phương tiện thông minh"""
    context = f"Thông tin văn bản:\n{text_context}\n\n"
    
    if relevant_images:
        context += "Thông tin hình ảnh có sẵn:\n"
        for i, img_info in enumerate(relevant_images, 1):
            context += f"\n[IMAGE_{i}]:"
            if img_info['alt']:
                context += f" Mô tả: {img_info['alt']}"
            if img_info['title']:
                context += f" Tiêu đề: {img_info['title']}"
            context += f" (Nguồn: {img_info['source_doc']['title']})"

    context += f"""

QUAN TRỌNG: Khi trả lời câu hỏi "{question}", hãy:
1. Đưa ra câu trả lời chi tiết và có cấu trúc
2. Khi đề cập đến hình ảnh, sử dụng cú pháp [IMAGE_X] để chỉ định ảnh nào cần hiển thị
3. Ví dụ: "Như bạn có thể thấy trong [IMAGE_1], điều này cho thấy..."
4. Sử dụng [IMAGE_X] ở những vị trí phù hợp trong câu trả lời để minh họa nội dung
5. Mỗi [IMAGE_X] sẽ được thay thế bằng hình ảnh tương ứng khi hiển thị

Hãy tham khảo cả thông tin văn bản và hình ảnh để đưa ra câu trả lời toàn diện và có minh họa phù hợp."""

    return context

def create_general_knowledge_context(question):
    """Tạo context cho kiến thức chung"""
    return f"""{question}"""

def parse_answer_with_image_markers(answer, relevant_images):
    """Phân tích câu trả lời và tách các marker ảnh"""
    image_pattern = r'\[IMAGE_(\d+)\]'
    parts = re.split(image_pattern, answer)
    
    content_parts = []
    for i, part in enumerate(parts):
        if part.isdigit():
            image_index = int(part) - 1
            if 0 <= image_index < len(relevant_images):
                content_parts.append({
                    'type': 'image',
                    'content': relevant_images[image_index],
                    'index': image_index
                })
        else:
            if part.strip():
                content_parts.append({
                    'type': 'text',
                    'content': part.strip()
                })
    
    return content_parts

def smart_display_answer_with_embedded_images(answer, relevant_images):
    """Hiển thị câu trả lời với ảnh được chèn thông minh"""
    if not relevant_images:
        st.markdown(answer)
        return

    content_parts = parse_answer_with_image_markers(answer, relevant_images)
    
    if not any(part['type'] == 'image' for part in content_parts):
        auto_embed_images_in_answer(answer, relevant_images)
        return

    for part in content_parts:
        if part['type'] == 'text':
            st.markdown(part['content'])
        elif part['type'] == 'image':
            img_info = part['content']
            try:
                image = Image.open(img_info['path'])
                caption = ""
                if img_info.get('alt'):
                    caption = f"📷 {img_info['alt']}"
                elif img_info.get('title'):
                    caption = f"📷 {img_info['title']}"
                else:
                    caption = f"📷 Hình ảnh từ {img_info['source_doc']['title']}"

                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")

def auto_embed_images_in_answer(answer, relevant_images):
    """Tự động chèn ảnh vào câu trả lời"""
    paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
    
    if len(paragraphs) <= 1:
        sentences = [s.strip() + '.' for s in answer.split('.') if s.strip()]
        paragraphs = sentences

    total_parts = len(paragraphs)
    total_images = len(relevant_images)

    if total_images == 0:
        st.markdown(answer)
        return

    insert_positions = []
    if total_parts > 1:
        step = max(1, total_parts // (total_images + 1))
        for i in range(min(total_images, total_parts - 1)):
            pos = (i + 1) * step
            if pos < total_parts:
                insert_positions.append(pos)

    image_index = 0
    for i, paragraph in enumerate(paragraphs):
        st.markdown(paragraph)
        
        if i in insert_positions and image_index < len(relevant_images):
            img_info = relevant_images[image_index]
            try:
                image = Image.open(img_info['path'])
                caption = ""
                if img_info.get('alt'):
                    caption = f"📷 {img_info['alt']}"
                elif img_info.get('title'):
                    caption = f"📷 {img_info['title']}"
                else:
                    caption = f"📷 Hình ảnh minh họa"

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                image_index += 1
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")

def collect_single_url(url):
    """Thu thập dữ liệu từ một URL"""
    try:
        scraper = WebScraperLocal(overwrite=False)
        success = scraper.fetch_and_save(url)
        return success, scraper.success_count, scraper.error_count
    except Exception as e:
        return False, 0, 1

def process_question(question):
    """Xử lý câu hỏi và hiển thị kết quả"""
    with st.spinner("🔍 Đang tìm kiếm và phân tích..."):
        try:
            # Khởi tạo AI client
            config = load_config_local()
            client = LocalGemmaClient(config["LOCAL_MODEL"])

            # Tìm kiếm documents liên quan
            relevant_docs, similarity = st.session_state.rag_system.enhanced_search_multimodal(
                question, threshold=0.3, top_k=3
            )

            # Kiểm tra xem có nên sử dụng kiến thức chung không
            use_general = should_use_general_knowledge(relevant_docs, similarity, question)

            if use_general:
                # Không tìm thấy thông tin liên quan - trả lời bằng kiến thức chung
                st.warning("⚠️ Không tìm thấy thông tin liên quan trong database. AI sẽ trả lời dựa trên kiến thức chung.")
                
                general_context = create_general_knowledge_context(question)
                
                answer = client.generate_response(
                    prompt=question,
                    context=general_context,
                    max_tokens=2000
                )

                # Lưu vào lịch sử với thông tin đặc biệt
                st.session_state.chat_history.append((question, answer, [], "general"))
                
                # Hiển thị kết quả
                st.markdown(f"**🙋 Bạn:** {question}")
                st.markdown("**🤖 AI (Kiến thức chung):**")
                st.markdown(answer)
                
                # Hiển thị thông tin debug
                with st.expander("🔍 Thông tin tìm kiếm"):
                    st.write(f"**Độ tương đồng:** {similarity:.3f} (thấp hơn ngưỡng)")
                    st.write(f"**Số documents tìm thấy:** {len(relevant_docs)}")
                    st.write("**Trạng thái:** Trả lời bằng kiến thức chung")

            else:
                # Tìm thấy thông tin liên quan - xử lý như bình thường
                # Tạo context từ documents
                text_context = ""
                for doc in relevant_docs:
                    text_context += f"\n\n--- {doc['title']} ---\n"
                    text_context += f"{doc['description']}\n"
                    text_context += f"{doc['text_content'][:1000]}..."

                # Lấy ảnh liên quan
                relevant_images = st.session_state.rag_system.get_relevant_images_for_context(
                    relevant_docs, question, max_images=5
                )

                # Tạo context đa phương tiện
                multimodal_context = create_intelligent_multimodal_context(
                    text_context, relevant_images, question
                )

                # Gọi AI để trả lời
                answer = client.generate_response(
                    prompt=question,
                    context=multimodal_context,
                    max_tokens=2000
                )

                # Lưu vào lịch sử
                st.session_state.chat_history.append((question, answer, relevant_images, "rag"))
                
                # Hiển thị kết quả
                st.markdown(f"**🙋 Bạn:** {question}")
                st.markdown("**🤖 AI:**")
                smart_display_answer_with_embedded_images(answer, relevant_images)
                
                # Hiển thị thông tin debug
                with st.expander("🔍 Thông tin tìm kiếm"):
                    st.write(f"**Độ tương đồng:** {similarity:.3f}")
                    st.write(f"**Số documents tìm thấy:** {len(relevant_docs)}")
                    st.write(f"**Số ảnh liên quan:** {len(relevant_images)}")
                    
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"**{i}.** {doc['title']}")

        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý câu hỏi: {e}")
            logger.error(f"Lỗi process_question: {e}")
            traceback.print_exc()

def main():
    st.set_page_config(
        page_title="🤖 Multimodal RAG Local với Gemma",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Khởi tạo session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedMultimodalRAGLocal("../db")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load config
    config = load_config_local()
    base_url = config["LOCAL_MODEL"]["base_url"]
    current_model = config["LOCAL_MODEL"]["model"]

    # Sidebar với URL Collector và thông tin Ollama
    with st.sidebar:
        st.markdown("## 🌐 Thu thập dữ liệu")
        
        # Ô nhập URL
        new_url = st.text_input(
            "Nhập URL để thu thập:",
            placeholder="https://example.com/article",
            help="Nhập URL của trang web bạn muốn thu thập dữ liệu"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Thu thập", use_container_width=True):
                if new_url.strip():
                    with st.spinner("Đang thu thập dữ liệu..."):
                        success, success_count, error_count = collect_single_url(new_url.strip())
                        
                        if success:
                            st.success(f"✅ Thu thập thành công!")
                            # Reload dữ liệu
                            st.session_state.rag_system.load_multimodal_data()
                            st.rerun()
                        else:
                            st.error(f"❌ Lỗi thu thập dữ liệu")
                else:
                    st.warning("⚠️ Vui lòng nhập URL")
        
        with col2:
            if st.button("🔄 Reload DB", use_container_width=True):
                with st.spinner("Đang tải lại dữ liệu..."):
                    st.session_state.rag_system.load_multimodal_data()
                    st.success("✅ Đã tải lại database")
                    st.rerun()

        st.markdown("---")

        # Thông tin Ollama Server
        st.markdown("## 🔗 Ollama Server")
        
        # Kiểm tra kết nối
        is_connected, status_msg, response_text = check_ollama_connection(base_url)
        
        if is_connected:
            st.success(f"✅ {status_msg}")
            st.caption(f"🌐 {base_url}")
        else:
            st.error(f"❌ {status_msg}")
            st.caption(f"🌐 {base_url}")
            
        # Hiển thị thông tin models
        if is_connected:
            with st.expander("🤖 Thông tin Models", expanded=True):
                # Model hiện tại
                st.markdown(f"**Model đang dùng:** `{current_model}`")
                
                # Lấy thông tin chi tiết model hiện tại
                success, model_info = get_model_info(base_url, current_model)
                if success:
                    if 'details' in model_info:
                        details = model_info['details']
                        st.write(f"**Kích thước:** {format_model_size(details.get('size'))}")
                        st.write(f"**Format:** {details.get('format', 'N/A')}")
                        st.write(f"**Family:** {details.get('family', 'N/A')}")
                        if 'parameter_size' in details:
                            st.write(f"**Parameters:** {details['parameter_size']}")
                
                # Models đang chạy
                success, running_models = get_running_models(base_url)
                if success and running_models:
                    st.markdown("**🟢 Models đang chạy:**")
                    for model in running_models:
                        name = model.get('name', 'Unknown')
                        size = format_model_size(model.get('size'))
                        expires_at = model.get('expires_at')
                        
                        st.write(f"• `{name}` ({size})")
                        if expires_at:
                            expire_time = format_duration(expires_at)
                            st.caption(f"  Expires: {expire_time}")
                else:
                    st.write("**🔴 Không có model nào đang chạy**")
                
                # Tất cả models có sẵn
                success, all_models = get_ollama_models(base_url)
                if success and all_models:
                    st.markdown("**📦 Models có sẵn:**")
                    for model in all_models[:5]:  # Hiển thị tối đa 5 models
                        name = model.get('name', 'Unknown')
                        size = format_model_size(model.get('size'))
                        modified = model.get('modified_at', '')
                        
                        if modified:
                            try:
                                mod_date = datetime.fromisoformat(modified.replace('Z', '+00:00'))
                                mod_str = mod_date.strftime('%d/%m/%Y')
                            except:
                                mod_str = modified[:10]
                        else:
                            mod_str = 'N/A'
                        
                        st.write(f"• `{name}` ({size})")
                        st.caption(f"  Modified: {mod_str}")
                    
                    if len(all_models) > 5:
                        st.caption(f"... và {len(all_models) - 5} models khác")

        st.markdown("---")

        # Thông tin Database
        st.markdown("## 📊 Thông tin Database")
        
        if st.session_state.rag_system.has_data:
            db_stats = st.session_state.rag_system.get_database_statistics()
            
            # Metrics tổng quan
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", db_stats['total_documents'])
                st.metric("Hình ảnh", db_stats['total_images'])
            with col2:
                text_kb = db_stats['total_text_length'] / 1024
                st.metric("Text", f"{text_kb:.1f} KB")
                
                # Tính tổng kích thước
                total_size = sum(site['size_bytes'] for site in db_stats['sites_info'])
                st.metric("Tổng kích thước", format_model_size(total_size))
            
            # Chi tiết từng site
            with st.expander("📚 Chi tiết Documents"):
                for i, site in enumerate(db_stats['sites_info'], 1):
                    st.write(f"**{i}.** {site['name']}")
                    st.caption(f"🔗 {site['url']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"🖼️ {site['images']} ảnh")
                    with col2:
                        st.caption(f"📄 {site['text_length']} ký tự")
                    with col3:
                        st.caption(f"💾 {format_model_size(site['size_bytes'])}")
                    
                    # Thời gian thu thập
                    scraped_at = site['scraped_at']
                    if scraped_at != 'N/A':
                        try:
                            scraped_date = datetime.fromisoformat(scraped_at.replace('Z', '+00:00'))
                            scraped_str = scraped_date.strftime('%d/%m/%Y %H:%M')
                            st.caption(f"⏰ Thu thập: {scraped_str}")
                        except:
                            st.caption(f"⏰ Thu thập: {scraped_at[:19]}")
                    
                    st.markdown("---")
        else:
            st.warning("⚠️ Chưa có dữ liệu trong database")
            st.info("💡 Hãy thu thập dữ liệu từ URL hoặc chạy collect_local.py")

        # Thống kê tìm kiếm
        stats = st.session_state.rag_system.get_search_statistics()
        if stats:
            st.markdown("### 📈 Thống kê Tìm kiếm")
            st.metric("Tổng truy vấn", stats['total_queries'])
            st.metric("Tỷ lệ thành công", f"{stats['success_rate']:.1f}%")
            st.metric("Độ tương đồng TB", f"{stats['average_similarity']:.3f}")
            
            # Intent distribution
            if stats['intent_distribution']:
                st.markdown("**Phân bố Intent:**")
                for intent, count in stats['intent_distribution'].items():
                    percentage = (count / stats['total_queries']) * 100
                    st.caption(f"• {intent}: {count} ({percentage:.1f}%)")

    # Main content
    st.title("🤖 Multimodal RAG Local với Gemma")
    st.markdown("*Hệ thống tìm kiếm và trả lời thông minh với hình ảnh*")

    # Hiển thị trạng thái kết nối
    col1, col2 = st.columns([3, 1])
    with col1:
        if is_connected:
            st.success(f"🔗 Kết nối Ollama: **{current_model}** tại {base_url}")
        else:
            st.error(f"❌ Không thể kết nối Ollama tại {base_url}")
    
    with col2:
        if st.button("🔄 Kiểm tra lại", use_container_width=True):
            st.rerun()

    # Thông báo về chế độ hoạt động
    if not st.session_state.rag_system.has_data:
        st.info("🧠 **Chế độ AI Chung**: Không có database, AI sẽ trả lời dựa trên kiến thức chung")
    else:
        st.success("🔍 **Chế độ RAG**: AI sẽ tìm kiếm trong database trước, nếu không có sẽ dùng kiến thức chung")

    # Chat interface
    st.markdown("## 💬 Trò chuyện với AI")

    # Hiển thị lịch sử chat
    for i, chat_item in enumerate(st.session_state.chat_history):
        if len(chat_item) == 4:  # Định dạng mới với mode
            question, answer, images, mode = chat_item
        else:  # Định dạng cũ
            question, answer, images = chat_item
            mode = "rag" if images else "general"
            
        with st.container():
            st.markdown(f"**🙋 Bạn:** {question}")
            if mode == "general":
                st.markdown("**🤖 AI (Kiến thức chung):**")
                st.markdown(answer)
            else:
                st.markdown("**🤖 AI:**")
                smart_display_answer_with_embedded_images(answer, images)
            st.markdown("---")

    # Input cho câu hỏi mới
    question = st.text_input(
        "Đặt câu hỏi của bạn:",
        placeholder="Ví dụ: Python là gì? Machine Learning hoạt động như thế nào?...",
        key="question_input"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Gửi câu hỏi", use_container_width=True) and question:
            if is_connected:
                process_question(question)
            else:
                st.error("❌ Không thể gửi câu hỏi. Vui lòng kiểm tra kết nối Ollama.")
    
    with col2:
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col3:
        threshold = st.slider("🎯 Ngưỡng tìm kiếm", 0.1, 0.8, 0.3, 0.1)

if __name__ == "__main__":
    main()
