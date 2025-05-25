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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_query_input(question, config):
    """Validate input query cơ bản"""
    validation_config = config.get('QUERY_VALIDATION', {})
    min_length = validation_config.get('min_query_length', 1)
    
    if len(question.strip()) < min_length:
        return False, f"Câu hỏi quá ngắn. Vui lòng nhập ít nhất {min_length} ký tự."
    
    return True, "Query hợp lệ"

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
        for site_dir in site_dirs:
            metadata_file = site_dir / "metadata.json"
            if not metadata_file.exists():
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
            except Exception as e:
                logger.error(f"Lỗi load dữ liệu từ {site_dir}: {e}")

        self.has_data = len(self.documents) > 0

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
            word_count = doc['metadata'].get('word_count', 0)
            db_info['total_words'] += word_count
            db_info['total_images'] += len(doc['images'])
            content_type = doc['metadata'].get('content_type', 'unknown')
            db_info['content_types'][content_type] = db_info['content_types'].get(content_type, 0) + 1
            db_info['titles'].append(doc['title'])
            db_info['urls'].append(doc['url'])
            if doc['text_content']:
                sample = doc['text_content'][:200] + "..."
                db_info['sample_content'].append({
                    'title': doc['title'],
                    'sample': sample
                })

        return db_info

    def enhanced_search_multimodal(self, query, threshold=None, top_k=3):
        """Tìm kiếm với threshold từ config"""
        if not self.has_data:
            return [], 0

        config = load_config_local()
        threshold = threshold or config.get('DB_THRESHOLD', 0.3)

        relevant_docs, max_similarity = enhanced_search_with_metadata(
            query, self.documents, threshold, top_k
        )

        if not relevant_docs and threshold > 0.2:
            logger.info(f"Không tìm thấy với threshold {threshold}, thử với 0.2")
            relevant_docs, max_similarity = enhanced_search_with_metadata(
                query, self.documents, 0.2, top_k
            )

        self.query_history.append({
            'query': query,
            'similarity': max_similarity,
            'results_count': len(relevant_docs),
            'timestamp': time.time(),
            'threshold_used': threshold
        })

        return relevant_docs, max_similarity

    def get_relevant_images_for_context(self, relevant_docs, query, max_images=5):
        """Lấy ảnh liên quan với scoring cải tiến"""
        relevant_images = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc in relevant_docs:
            for img_info in doc['images']:
                relevance_score = 0
                alt_text = img_info.get('alt', '').lower()
                title_text = img_info.get('title', '').lower()

                alt_words = set(alt_text.split())
                title_words = set(title_text.split())

                alt_matches = len(query_words.intersection(alt_words))
                title_matches = len(query_words.intersection(title_words))
                relevance_score += (alt_matches * 2) + (title_matches * 2)

                for word in query_words:
                    if len(word) > 3:
                        if word in alt_text:
                            relevance_score += 1
                        if word in title_text:
                            relevance_score += 1

                if len(alt_text) > 20 or len(title_text) > 20:
                    relevance_score += 0.5

                img_path = Path(img_info.get('local_path', ''))
                if img_path.exists() and (relevance_score > 0 or (not alt_text and not title_text)):
                    relevant_images.append({
                        'path': img_path,
                        'alt': img_info.get('alt', ''),
                        'title': img_info.get('title', ''),
                        'url': img_info.get('url', ''),
                        'source_doc': doc,
                        'relevance_score': relevance_score
                    })

        unique_images = []
        seen_paths = set()
        for img in sorted(relevant_images, key=lambda x: x['relevance_score'], reverse=True):
            if img['path'] not in seen_paths:
                unique_images.append(img)
                seen_paths.add(img['path'])

        return unique_images[:max_images]

def display_database_debug_info(rag_system):
    """Hiển thị thông tin debug chi tiết về database"""
    st.subheader("🔍 Thông tin chi tiết Database")
    db_info = rag_system.get_database_info()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", db_info['total_documents'])
    with col2:
        st.metric("Tổng từ", f"{db_info['total_words']:,}")
    with col3:
        st.metric("Tổng ảnh", db_info['total_images'])
    with col4:
        st.metric("Loại nội dung", len(db_info['content_types']))

    if db_info['content_types']:
        st.write("**Phân loại nội dung:**")
        for content_type, count in db_info['content_types'].items():
            st.write(f"- {content_type}: {count} documents")

    with st.expander("📋 Danh sách Documents", expanded=False):
        for i, title in enumerate(db_info['titles'], 1):
            st.write(f"{i}. **{title}**")
            if i-1 < len(db_info['urls']):
                st.caption(f"URL: {db_info['urls'][i-1]}")

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
        server_info['status'] = 'Connected' if connection_status else 'Connection Failed'

    except Exception as e:
        server_info['status'] = f'Error: {str(e)}'
        logger.error(f"Lỗi khi lấy thông tin server: {e}")

    return server_info

def display_server_info():
    """Hiển thị thông tin server trong sidebar"""
    config = load_config_local()
    server_info = get_server_info(config)

    st.sidebar.header("🖥️ Thông tin Server")
    status = server_info['status']
    if server_info['connection_status']:
        st.sidebar.success(f"✅ {status}")
    elif 'Error' in status or 'Failed' in status:
        st.sidebar.error(f"❌ {status}")
    else:
        st.sidebar.warning(f"⚠️ {status}")

    with st.sidebar.expander("Chi tiết Server", expanded=False):
        st.write(f"**URL:** {server_info['base_url']}")
        if server_info['response_time']:
            st.write(f"**Thời gian phản hồi:** {server_info['response_time']:.3f}s")
        st.write(f"**Model hiện tại:** {server_info['model']}")

def create_sidebar():
    """Tạo sidebar với các chức năng điều khiển"""
    st.sidebar.title("🔧 Điều khiển hệ thống")
    display_server_info()

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
    """Xử lý fallback khi không tìm thấy kết quả"""
    st.info("🔍 Không tìm thấy thông tin liên quan trong database. Đang gọi AI để trả lời...")
    
    fallback_context = f"""
Câu hỏi: {question}

Thông tin: Không tìm thấy thông tin cụ thể trong cơ sở dữ liệu.

Hướng dẫn trả lời:
1. Trả lời bằng tiếng Việt
2. Dựa trên kiến thức chung
3. Nếu không biết, hãy nói rõ và đề xuất tìm kiếm thêm
4. Trả lời chi tiết và hữu ích

Trả lời:
"""
    
    try:
        response = ask_local_model(question, fallback_context, config)
        st.info("💡 **Lưu ý:** Câu trả lời dựa trên kiến thức chung của AI.")
        return response
    except Exception as e:
        logger.error(f"Error in fallback AI response: {e}")
        return f"Xin lỗi, không thể trả lời do lỗi: {e}"

def create_intelligent_multimodal_context(text_context, relevant_images, question):
    """Tạo context đa phương tiện thông minh"""
    context = f"Thông tin văn bản:\n{text_context}\n\n"
    
    if relevant_images:
        context += "Thông tin hình ảnh:\n"
        for i, img_info in enumerate(relevant_images, 1):
            context += f"\n[IMAGE_{i}]:"
            if img_info['alt']:
                context += f" Mô tả: {img_info['alt']}"
            if img_info['title']:
                context += f" Tiêu đề: {img_info['title']}"
            context += f" (Nguồn: {img_info['source_doc']['title']})"

    context += f"""

Hướng dẫn trả lời:
1. Trả lời chi tiết bằng tiếng Việt
2. Sử dụng [IMAGE_X] để chỉ định ảnh minh họa
3. Dựa trên thông tin văn bản và hình ảnh
4. Nếu không có thông tin liên quan, nói rõ
5. Trả lời câu hỏi: {question}
"""
    return context

def parse_answer_with_image_markers(answer, relevant_images):
    """Phân tích câu trả lời và tách marker ảnh"""
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
    """Hiển thị câu trả lời với ảnh chèn thông minh"""
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
                caption = img_info.get('alt') or img_info.get('title') or f"Hình ảnh từ {img_info['source_doc']['title']}"
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                st.markdown("", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")

def auto_embed_images_in_answer(answer, relevant_images):
    """Tự động chèn ảnh vào câu trả lời"""
    paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
    total_images = len(relevant_images)

    if total_images == 0:
        st.markdown(answer)
        return

    insert_positions = []
    if len(paragraphs) > 1:
        step = max(1, len(paragraphs) // (total_images + 1))
        for i in range(min(total_images, len(paragraphs) - 1)):
            pos = (i + 1) * step
            if pos < len(paragraphs):
                insert_positions.append(pos)

    image_index = 0
    for i, paragraph in enumerate(paragraphs):
        st.markdown(paragraph)
        if i in insert_positions and image_index < len(relevant_images):
            img_info = relevant_images[image_index]
            try:
                image = Image.open(img_info['path'])
                caption = img_info.get('alt') or img_info.get('title') or "Hình ảnh minh họa"
                st.markdown("", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                st.markdown("", unsafe_allow_html=True)
                image_index += 1
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")

def main():
    """Hàm chính của ứng dụng"""
    st.set_page_config(
        page_title="RAG Local với Ảnh Chèn",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    create_sidebar()

    st.title("🤖 Hệ thống RAG Local với Ảnh Chèn Thông Minh")
    st.markdown("---")

    if 'rag_system' not in st.session_state:
        with st.spinner("Đang khởi tạo hệ thống..."):
            st.session_state.rag_system = EnhancedMultimodalRAGLocal("db")

    rag_system = st.session_state.rag_system

    if not rag_system.has_data:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng thu thập dữ liệu từ sidebar.")
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **Hướng dẫn sử dụng:**")
            st.write("1. Sử dụng sidebar để thêm URL")
            st.write("2. Nhấn 'Thu thập' để tải dữ liệu")
            st.write("3. Đặt câu hỏi sau khi có dữ liệu")
        with col2:
            st.info("🔧 **Tính năng:**")
            st.write("- Thu thập từ URL")
            st.write("- Ảnh chèn thông minh")
            st.write("- AI trả lời mọi câu hỏi")

    if rag_system.has_data:
        display_database_debug_info(rag_system)
        st.markdown("---")

    st.header("💬 Hỏi đáp thông minh với AI")
    question = st.text_input(
        "Đặt câu hỏi:",
        placeholder="Bạn có thể hỏi bất kỳ điều gì...",
        help="Nhập câu hỏi và nhấn Enter."
    )

    with st.expander("⚙️ Tùy chọn tìm kiếm", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider(
                "Ngưỡng tìm kiếm:",
                min_value=0.3,
                max_value=0.5,
                value=0.3,
                step=0.05
            )
        with col2:
            max_results = st.selectbox(
                "Số kết quả tối đa:",
                options=[1, 2, 3, 4, 5],
                index=2
            )

    if question:
        config = load_config_local()
        is_valid, validation_message = validate_query_input(question, config)
        if not is_valid:
            st.error(f"❌ {validation_message}")
            return

        with st.spinner("🔍 Đang tìm kiếm và tạo câu trả lời..."):
            try:
                st.info(f"🔍 Đang tìm kiếm: '{question}' với threshold {threshold}")

                relevant_docs, similarity = rag_system.enhanced_search_multimodal(
                    question, threshold, max_results
                )

                st.subheader("🔍 Kết quả tìm kiếm")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Độ tương tự tối đa", f"{similarity:.3f}")
                with col2:
                    st.metric("Số documents", len(relevant_docs))
                with col3:
                    st.metric("Ngưỡng sử dụng", f"{threshold:.3f}")
                with col4:
                    st.metric("Ngưỡng tối thiểu", f"{config.get('MIN_SIMILARITY_THRESHOLD', 0.5):.3f}")

                if relevant_docs and similarity >= config.get('MIN_SIMILARITY_THRESHOLD', 0.5):
                    with st.expander("📋 Documents được tìm thấy", expanded=True):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.write(f"**{i}. {doc['title']}**")
                            st.caption(f"URL: {doc['url']}")
                            st.caption(f"Loại: {doc['metadata'].get('content_type', 'unknown')}")
                            snippet = doc['text_content'][:300] + "..."
                            highlighted_snippet = snippet
                            for word in question.split():
                                if len(word) > 3:
                                    highlighted_snippet = highlighted_snippet.replace(
                                        word, f"**{word}**"
                                    )
                            st.write(f"*Đoạn trích:* {highlighted_snippet}")
                            st.write("---")

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

                    answer = ask_local_model(question, context, config)

                    st.subheader("📝 Câu trả lời:")
                    st.success("✅ **Dựa trên database**")
                    smart_display_answer_with_embedded_images(answer, relevant_images)

                    if relevant_images:
                        with st.expander("🖼️ Thông tin ảnh", expanded=False):
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
                    st.subheader("📝 Câu trả lời:")
                    fallback_answer = handle_no_results_fallback(question, config)
                    st.markdown(fallback_answer)

            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý câu hỏi: {e}")
                logger.error(f"Lỗi: {e}")

                st.info("🔄 Đang thử phương pháp dự phòng...")
                try:
                    emergency_answer = handle_no_results_fallback(question, config)
                    st.subheader("📝 Câu trả lời (Dự phòng):")
                    st.markdown(emergency_answer)
                except Exception as e2:
                    st.error(f"❌ Không thể trả lời: {e2}")

    st.markdown("---")
    st.markdown("🤖 **RAG Local System** - Trả lời dựa trên database hoặc kiến thức chung")

if __name__ == "__main__":
    main()