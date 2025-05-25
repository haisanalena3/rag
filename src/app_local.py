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
import time
import shutil
import requests

# Fix torch classes path issue with Streamlit on Python 3.13
try:
    import torch
    if not hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not fix torch.classes path: {e}")

from config_local import load_config_local
from rag_local import (
    load_documents, search_documents, ask_local_model,
    search_documents_with_threshold, adaptive_threshold_search,
    enhanced_search_with_metadata
)
from collect_local import WebScraperLocal
from local_gemma import LocalGemmaClient

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
        """Load dữ liệu đa phương tiện từ database với debug chi tiết"""
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

    def get_database_info(self):
        """Lấy thông tin chi tiết về database"""
        db_info = {
            'total_documents': len(self.documents),
            'total_words': 0,
            'total_images': 0,
            'content_types': {},
            'keywords': set(),
            'titles': [],
            'urls': [],
            'sample_content': []
        }

        for doc in self.documents:
            # Thống kê cơ bản
            word_count = doc['metadata'].get('word_count', 0)
            db_info['total_words'] += word_count
            db_info['total_images'] += len(doc['images'])
            
            # Content types
            content_type = doc['metadata'].get('content_type', 'unknown')
            db_info['content_types'][content_type] = db_info['content_types'].get(content_type, 0) + 1
            
            # Titles và URLs
            db_info['titles'].append(doc['title'])
            db_info['urls'].append(doc['url'])
            
            # Sample content (first 200 chars)
            if doc['text_content']:
                sample = doc['text_content'][:200] + "..."
                db_info['sample_content'].append({
                    'title': doc['title'],
                    'sample': sample
                })

        return db_info

    def enhanced_search_multimodal(self, query, threshold=0.05, top_k=3):
        """Tìm kiếm với threshold thấp hơn để tìm được kết quả"""
        if not self.has_data:
            return [], 0

        # Thử với enhanced_search_with_metadata với threshold thấp
        relevant_docs, max_similarity = enhanced_search_with_metadata(
            query, self.documents, threshold, top_k
        )
        
        # Nếu không tìm thấy, thử với threshold cực thấp
        if not relevant_docs and threshold > 0.01:
            logger.info(f"Không tìm thấy với threshold {threshold}, thử với 0.01")
            relevant_docs, max_similarity = enhanced_search_with_metadata(
                query, self.documents, 0.01, top_k
            )
        
        # Nếu vẫn không tìm thấy, thử tìm kiếm từng từ
        if not relevant_docs:
            logger.info(f"Thử tìm kiếm từng từ trong query: {query}")
            words = query.split()
            for word in words:
                if len(word) > 2:  # Bỏ qua từ quá ngắn
                    word_results, word_similarity = enhanced_search_with_metadata(
                        word, self.documents, 0.01, top_k
                    )
                    if word_results:
                        relevant_docs = word_results
                        max_similarity = word_similarity
                        logger.info(f"Tìm thấy kết quả với từ: {word}")
                        break

        # Lưu lịch sử
        self.query_history.append({
            'query': query,
            'similarity': max_similarity,
            'results_count': len(relevant_docs),
            'timestamp': time.time(),
            'threshold_used': threshold
        })

        return relevant_docs, max_similarity

    def get_relevant_images_for_context(self, relevant_docs, query, max_images=5):
        """Lấy ảnh liên quan để đưa vào context với scoring cải tiến"""
        relevant_images = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc in relevant_docs:
            for img_info in doc['images']:
                relevance_score = 0
                alt_text = img_info.get('alt', '').lower()
                title_text = img_info.get('title', '').lower()

                # Tính điểm dựa trên từ khóa
                alt_words = set(alt_text.split())
                title_words = set(title_text.split())

                # Exact word matches
                alt_matches = len(query_words.intersection(alt_words))
                title_matches = len(query_words.intersection(title_words))
                relevance_score += (alt_matches * 2) + (title_matches * 2)

                # Partial matches
                for word in query_words:
                    if len(word) > 3:
                        if word in alt_text:
                            relevance_score += 1
                        if word in title_text:
                            relevance_score += 1

                # Bonus điểm cho ảnh có mô tả chi tiết
                if len(alt_text) > 20 or len(title_text) > 20:
                    relevance_score += 0.5

                # Bonus cho ảnh từ document có điểm cao
                if hasattr(doc, 'search_weight'):
                    relevance_score *= doc.search_weight

                # Thêm ảnh nếu có điểm hoặc không có mô tả (fallback)
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

        # Sắp xếp theo điểm và loại bỏ trùng lặp
        seen_paths = set()
        unique_images = []
        for img in sorted(relevant_images, key=lambda x: x['relevance_score'], reverse=True):
            if img['path'] not in seen_paths:
                unique_images.append(img)
                seen_paths.add(img['path'])

        return unique_images[:max_images]

def display_database_debug_info(rag_system):
    """Hiển thị thông tin debug chi tiết về database"""
    st.subheader("🔍 Thông tin chi tiết Database")
    
    db_info = rag_system.get_database_info()
    
    # Metrics tổng quan
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", db_info['total_documents'])
    with col2:
        st.metric("Tổng từ", f"{db_info['total_words']:,}")
    with col3:
        st.metric("Tổng ảnh", db_info['total_images'])
    with col4:
        st.metric("Loại nội dung", len(db_info['content_types']))

    # Chi tiết content types
    if db_info['content_types']:
        st.write("**Phân loại nội dung:**")
        for content_type, count in db_info['content_types'].items():
            st.write(f"- {content_type}: {count} documents")

    # Danh sách documents
    with st.expander("📋 Danh sách Documents trong Database", expanded=False):
        for i, title in enumerate(db_info['titles'], 1):
            st.write(f"{i}. **{title}**")
            if i-1 < len(db_info['urls']):
                st.caption(f"URL: {db_info['urls'][i-1]}")

    # Sample content
    with st.expander("📄 Mẫu nội dung", expanded=False):
        for sample in db_info['sample_content'][:3]:
            st.write(f"**{sample['title']}**")
            st.write(sample['sample'])
            st.write("---")

def get_server_info(config):
    """Lấy thông tin server model API"""
    server_info = {
        'status': 'unknown',
        'base_url': config.get('LOCAL_MODEL', {}).get('base_url', 'N/A'),
        'model': config.get('LOCAL_MODEL', {}).get('model', 'N/A'),
        'connection_status': False,
        'response_time': None
    }

    try:
        base_url = server_info['base_url']
        if not base_url or base_url == 'N/A':
            server_info['status'] = 'No URL configured'
            return server_info

        local_client = LocalGemmaClient(base_url=base_url, model=server_info['model'])
        start_time = time.time()
        connection_status = local_client.check_connection()
        response_time = time.time() - start_time

        server_info['connection_status'] = connection_status
        server_info['response_time'] = response_time

        if connection_status:
            server_info['status'] = 'Connected'
        else:
            server_info['status'] = 'Connection Failed'

    except Exception as e:
        server_info['status'] = f'Error: {str(e)}'
        logger.error(f"Lỗi khi lấy thông tin server: {e}")

    return server_info

def display_server_info():
    """Hiển thị thông tin server trong sidebar"""
    config = load_config_local()
    server_info = get_server_info(config)

    st.sidebar.header("🖥️ Thông tin Server")

    # Status indicator
    status = server_info['status']
    if server_info['connection_status']:
        st.sidebar.success(f"✅ {status}")
    elif 'Error' in status or 'Failed' in status:
        st.sidebar.error(f"❌ {status}")
    else:
        st.sidebar.warning(f"⚠️ {status}")

    # Server details
    with st.sidebar.expander("Chi tiết Server", expanded=False):
        st.write(f"**URL:** {server_info['base_url']}")
        if server_info['response_time']:
            st.write(f"**Thời gian phản hồi:** {server_info['response_time']:.3f}s")

    # Model info
    st.write(f"**Model hiện tại:** {server_info['model']}")

def create_sidebar():
    """Tạo sidebar với các chức năng điều khiển"""
    st.sidebar.title("🔧 Điều khiển hệ thống")
    display_server_info()

    # Phần nhập URL mới
    st.sidebar.header("📥 Thu thập dữ liệu mới")
    with st.sidebar.expander("Thêm URL mới", expanded=False):
        new_url = st.text_input(
            "Nhập URL cần thu thập:",
            placeholder="https://example.com/article",
            key="new_url_input"
        )

        if st.button("Thu thập", key="collect_single", type="primary"):
            if new_url:
                collect_single_url(new_url)
            else:
                st.error("Vui lòng nhập URL")

def collect_single_url(url):
    """Thu thập dữ liệu từ một URL"""
    try:
        if not url.startswith(('http://', 'https://')):
            st.error("URL phải bắt đầu bằng http:// hoặc https://")
            return

        with st.spinner(f"Đang thu thập từ {url}..."):
            scraper = WebScraperLocal(overwrite=False)
            success = scraper.fetch_and_save(url)
            
            if success:
                st.success(f"✅ Thu thập thành công từ {url}")
                if 'rag_system' in st.session_state:
                    st.session_state.rag_system.load_multimodal_data()
                st.rerun()
            else:
                st.error(f"❌ Không thể thu thập từ {url}")
    except Exception as e:
        st.error(f"Lỗi: {e}")
        logger.error(f"Lỗi thu thập {url}: {e}")

def handle_no_results_fallback(question, config):
    """Xử lý fallback khi không tìm thấy kết quả trong database"""
    st.info("🔍 Không tìm thấy thông tin liên quan trong database. Đang gọi AI để trả lời...")
    
    fallback_context = f"""
    Câu hỏi: {question}
    
    Thông tin: Không tìm thấy thông tin cụ thể trong cơ sở dữ liệu về câu hỏi này.
    
    Hướng dẫn trả lời:
    1. Trả lời dựa trên kiến thức chung về chủ đề được hỏi
    2. Nêu rõ rằng đây là câu trả lời chung, không dựa trên dữ liệu cụ thể
    3. Đề xuất người dùng tìm kiếm thêm thông tin hoặc cung cấp thêm dữ liệu
    4. Trả lời bằng tiếng Việt một cách chi tiết và hữu ích
    """
    
    try:
        response = ask_local_model(question, fallback_context, config)
        st.warning("⚠️ Câu trả lời dưới đây dựa trên kiến thức chung của AI, không dựa trên dữ liệu trong database của bạn.")
        return response
    except Exception as e:
        logger.error(f"Error in fallback AI response: {e}")
        return f"Xin lỗi, tôi không thể trả lời câu hỏi này vì không tìm thấy thông tin liên quan trong database và không thể kết nối với AI."

def create_intelligent_multimodal_context(text_context, relevant_images, question):
    """Tạo context đa phương tiện thông minh với hướng dẫn chèn ảnh"""
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
            if img_info['relevance_score'] > 0:
                context += f" (Độ liên quan: {img_info['relevance_score']:.1f})"

    context += f"""

QUAN TRỌNG: Khi trả lời câu hỏi "{question}", hãy:
1. Đưa ra câu trả lời chi tiết và có cấu trúc
2. Khi đề cập đến hình ảnh, sử dụng cú pháp [IMAGE_X] để chỉ định ảnh nào cần hiển thị
3. Ví dụ: "Như bạn có thể thấy trong [IMAGE_1], điều này cho thấy..."
4. Sử dụng [IMAGE_X] ở những vị trí phù hợp trong câu trả lời để minh họa nội dung
5. Mỗi [IMAGE_X] sẽ được thay thế bằng hình ảnh tương ứng khi hiển thị
6. Trả lời bằng tiếng Việt và dựa trên thông tin đã cung cấp

Hãy tham khảo cả thông tin văn bản và hình ảnh để đưa ra câu trả lời toàn diện và có minh họa phù hợp."""
    
    return context

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
        # Nếu AI không sử dụng [IMAGE_X], tự động chèn ảnh
        auto_embed_images_in_answer(answer, relevant_images)
        return

    # Hiển thị theo thứ tự AI đã chỉ định
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

                # Hiển thị ảnh với caption
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                st.markdown("", unsafe_allow_html=True)  # Spacing
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")

def auto_embed_images_in_answer(answer, relevant_images):
    """Tự động chèn ảnh vào câu trả lời nếu AI không sử dụng [IMAGE_X]"""
    paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
    
    if len(paragraphs) <= 1:
        # Nếu không có đoạn văn, chia theo câu
        sentences = [s.strip() + '.' for s in answer.split('.') if s.strip()]
        paragraphs = sentences

    total_parts = len(paragraphs)
    total_images = len(relevant_images)

    if total_images == 0:
        st.markdown(answer)
        return

    # Tính toán vị trí chèn ảnh
    insert_positions = []
    if total_parts > 1:
        step = max(1, total_parts // (total_images + 1))
        for i in range(min(total_images, total_parts - 1)):
            pos = (i + 1) * step
            if pos < total_parts:
                insert_positions.append(pos)

    # Hiển thị với ảnh được chèn
    image_index = 0
    for i, paragraph in enumerate(paragraphs):
        st.markdown(paragraph)
        
        # Chèn ảnh tại vị trí được tính toán
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

                # Spacing trước ảnh
                st.markdown("", unsafe_allow_html=True)
                
                # Hiển thị ảnh ở giữa
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                
                # Spacing sau ảnh
                st.markdown("", unsafe_allow_html=True)
                image_index += 1
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")

def main():
    """Hàm chính của ứng dụng với ảnh chèn trong nội dung"""
    st.set_page_config(
        page_title="RAG Local với Ảnh Chèn",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Tạo sidebar
    create_sidebar()

    # Header chính
    st.title("🤖 Hệ thống RAG Local với Ảnh Chèn Thông Minh")
    st.markdown("---")

    # Khởi tạo RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Đang khởi tạo hệ thống..."):
            st.session_state.rag_system = EnhancedMultimodalRAGLocal("db")

    rag_system = st.session_state.rag_system

    # Kiểm tra dữ liệu
    if not rag_system.has_data:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng thu thập dữ liệu từ sidebar.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **Hướng dẫn sử dụng:**")
            st.write("1. Sử dụng sidebar bên trái để thêm URL")
            st.write("2. Nhấn 'Thu thập' để tải dữ liệu")
            st.write("3. Sau khi có dữ liệu, bạn có thể đặt câu hỏi")
        
        with col2:
            st.info("🔧 **Tính năng có sẵn:**")
            st.write("- Thu thập từ URL đơn lẻ")
            st.write("- Ảnh chèn thông minh trong câu trả lời")
            st.write("- AI fallback khi không tìm thấy")
            st.write("- Debug thông tin database")
        
        return

    # Hiển thị thông tin database
    display_database_debug_info(rag_system)
    st.markdown("---")

    # Phần hỏi đáp chính
    st.header("💬 Hỏi đáp thông minh với Ảnh Minh Họa")

    # Input câu hỏi
    question = st.text_input(
        "Đặt câu hỏi:",
        placeholder="Ví dụ: cách cài đặt Oracle 19c, ZooKeeper là gì...",
        help="Nhập câu hỏi và nhấn Enter để tìm kiếm"
    )

    # Tùy chọn nâng cao
    with st.expander("⚙️ Tùy chọn tìm kiếm", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            threshold = st.slider(
                "Ngưỡng tìm kiếm:",
                min_value=0.01,
                max_value=0.8,
                value=0.05,
                step=0.01,
                help="Ngưỡng thấp hơn = kết quả nhiều hơn nhưng ít chính xác hơn"
            )
        
        with col2:
            max_results = st.selectbox(
                "Số kết quả tối đa:",
                options=[1, 2, 3, 4, 5],
                index=2
            )

    if question:
        with st.spinner("🔍 Đang tìm kiếm và tạo câu trả lời..."):
            try:
                # Hiển thị debug info
                st.info(f"🔍 Đang tìm kiếm: '{question}' với threshold {threshold}")
                
                # Tìm kiếm documents liên quan
                relevant_docs, similarity = rag_system.enhanced_search_multimodal(
                    question, threshold, max_results
                )

                # Debug thông tin tìm kiếm chi tiết
                st.subheader("🔍 Kết quả tìm kiếm")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Độ tương tự tối đa", f"{similarity:.3f}")
                with col2:
                    st.metric("Số documents tìm thấy", len(relevant_docs))
                with col3:
                    st.metric("Ngưỡng sử dụng", f"{threshold:.3f}")
                with col4:
                    # Hiển thị threshold đề xuất
                    suggested_threshold = max(0.01, similarity * 0.8) if similarity > 0 else 0.01
                    st.metric("Threshold đề xuất", f"{suggested_threshold:.3f}")

                # Nếu không tìm thấy, thử các phương pháp khác
                if not relevant_docs:
                    st.warning("⚠️ Không tìm thấy với tìm kiếm chính, đang thử các phương pháp khác...")
                    
                    # Thử tìm kiếm fuzzy
                    words = question.split()
                    st.write("**Đang thử tìm kiếm từng từ:**")
                    for word in words:
                        if len(word) > 2:
                            word_results, word_sim = rag_system.enhanced_search_multimodal(word, 0.01, 1)
                            st.write(f"- '{word}': {len(word_results)} kết quả (sim: {word_sim:.3f})")
                            if word_results:
                                relevant_docs = word_results
                                similarity = word_sim
                                st.success(f"✅ Tìm thấy kết quả với từ khóa: '{word}'")
                                break

                if relevant_docs:
                    # Hiển thị documents tìm thấy
                    with st.expander("📋 Documents được tìm thấy", expanded=True):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.write(f"**{i}. {doc['title']}**")
                            st.caption(f"URL: {doc['url']}")
                            st.caption(f"Loại: {doc['metadata'].get('content_type', 'unknown')}")
                            st.caption(f"Số từ: {doc['metadata'].get('word_count', 0)}")
                            
                            # Show snippet với highlight
                            snippet = doc['text_content'][:300] + "..."
                            # Highlight query words trong snippet
                            highlighted_snippet = snippet
                            for word in question.split():
                                if len(word) > 3:
                                    highlighted_snippet = highlighted_snippet.replace(
                                        word, f"**{word}**"
                                    )
                            st.write(f"*Đoạn trích:* {highlighted_snippet}")
                            st.write("---")

                    # Tạo context và trả lời
                    text_context = "\n\n".join([
                        f"Tiêu đề: {doc['title']}\nMô tả: {doc['description']}\nNội dung: {doc['text_content'][:1000]}..."
                        for doc in relevant_docs
                    ])

                    relevant_images = rag_system.get_relevant_images_for_context(
                        relevant_docs, question, 5
                    )

                    context = create_intelligent_multimodal_context(
                        text_context, relevant_images, question
                    )

                    config = load_config_local()
                    answer = ask_local_model(question, context, config)

                    # Hiển thị kết quả với ảnh chèn thông minh
                    st.subheader("📝 Câu trả lời:")
                    smart_display_answer_with_embedded_images(answer, relevant_images)

                    # Hiển thị thông tin ảnh đã sử dụng
                    if relevant_images:
                        with st.expander("🖼️ Thông tin ảnh đã sử dụng", expanded=False):
                            for i, img_info in enumerate(relevant_images, 1):
                                st.write(f"**Ảnh {i}:**")
                                st.write(f"- Nguồn: {img_info['source_doc']['title']}")
                                if img_info['alt']:
                                    st.write(f"- Mô tả: {img_info['alt']}")
                                if img_info['title']:
                                    st.write(f"- Tiêu đề: {img_info['title']}")
                                st.write(f"- Độ liên quan: {img_info['relevance_score']:.2f}")
                                st.write("---")

                else:
                    # Fallback mechanism - gọi AI trực tiếp
                    st.subheader("📝 Câu trả lời (AI Fallback):")
                    config = load_config_local()
                    fallback_answer = handle_no_results_fallback(question, config)
                    st.markdown(fallback_answer)
                    
                    st.info("💡 **Gợi ý cải thiện:**")
                    st.write("- Database có thể chưa chứa thông tin về chủ đề này")
                    st.write("- Thử sử dụng từ khóa đơn giản hơn")
                    st.write("- Thu thập thêm dữ liệu liên quan từ sidebar")

            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý câu hỏi: {e}")
                logger.error(f"Lỗi xử lý câu hỏi: {e}")
                
                # Emergency fallback
                st.info("🔄 Đang thử phương pháp dự phòng...")
                try:
                    config = load_config_local()
                    emergency_answer = handle_no_results_fallback(question, config)
                    st.subheader("📝 Câu trả lời (Dự phòng):")
                    st.markdown(emergency_answer)
                except Exception as e2:
                    st.error(f"❌ Không thể tạo câu trả lời: {e2}")
                    st.info("Vui lòng thử lại sau hoặc kiểm tra kết nối với model local.")

    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 **RAG Local System với Ảnh Chèn Thông Minh** - "
        "Powered by Enhanced Search & Streamlit"
    )

if __name__ == "__main__":
    main()
