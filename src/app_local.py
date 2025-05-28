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
import hashlib
from datetime import datetime
import asyncio
import nest_asyncio

from config_local import load_config_local
from rag_local import (
    load_documents, search_documents, ask_local_model,
    search_documents_with_threshold, adaptive_threshold_search,
    enhanced_search_with_metadata, get_vector_database
)
from collect_local import WebScraperLocal
from text_collector import TextContentCollector
from local_gemma import LocalGemmaClient
from mcp_client import MCPClient

# Áp dụng nest_asyncio để hỗ trợ asyncio trong Streamlit
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def get_available_models():
    """Lấy danh sách models có sẵn từ Ollama server"""
    try:
        config = load_config_local()
        base_url = config.get('LOCAL_MODEL', {}).get('base_url', 'http://localhost:11434')
        
        local_client = LocalGemmaClient(base_url=base_url)
        available_models = local_client.get_available_models()
        
        if available_models:
            return available_models
        else:
            return ["qwen2.5:3b"]
            
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách models: {e}")
        return ["qwen2.5:3b"]

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

def get_server_info(config, selected_model=None):
    """Lấy thông tin server model API với model được chọn"""
    server_info = {
        'status': 'unknown',
        'base_url': config.get('LOCAL_MODEL', {}).get('base_url', 'N/A'),
        'model': selected_model or config.get('LOCAL_MODEL', {}).get('model', 'N/A'),
        'connection_status': False,
        'response_time': None,
        'available_models': []
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
            available_models = local_client.get_available_models()
            server_info['available_models'] = available_models
            server_info['status'] = 'Connected'
            
            if selected_model and not any(selected_model in model for model in available_models):
                server_info['status'] = f'Model {selected_model} not found'
        else:
            server_info['status'] = 'Connection Failed'
            
    except Exception as e:
        server_info['status'] = f'Error: {str(e)}'
        logger.error(f"Lỗi khi lấy thông tin server: {e}")
    
    return server_info

def display_server_info(selected_model=None):
    """Hiển thị thông tin server trong sidebar với model được chọn"""
    config = load_config_local()
    server_info = get_server_info(config, selected_model)
    
    st.sidebar.header("🖥️ Thông tin Server")
    
    status = server_info['status']
    if server_info['connection_status']:
        st.sidebar.success(f"✅ {status}")
    elif 'Error' in status or 'Failed' in status or 'not found' in status:
        st.sidebar.error(f"❌ {status}")
    else:
        st.sidebar.warning(f"⚠️ {status}")
    
    with st.sidebar.expander("Chi tiết Server", expanded=False):
        st.write(f"**URL:** {server_info['base_url']}")
        if server_info['response_time']:
            st.write(f"**Thời gian phản hồi:** {server_info['response_time']:.3f}s")
        st.write(f"**Model hiện tại:** {server_info['model']}")
        
        if server_info['available_models']:
            st.write("**Models có sẵn:**")
            for model in server_info['available_models']:
                if model == server_info['model']:
                    st.write(f"- ✅ **{model}** (đang sử dụng)")
                else:
                    st.write(f"- {model}")

def display_api_monitor():
    """Hiển thị monitor API requests"""
    with st.sidebar.expander("📊 API Monitor", expanded=False):
        if 'api_requests' not in st.session_state:
            st.session_state.api_requests = []
        
        st.write("**Lịch sử API Requests:**")
        if st.session_state.api_requests:
            for i, req in enumerate(reversed(st.session_state.api_requests[-5:]), 1):
                with st.container():
                    st.write(f"**Request {i}:**")
                    st.caption(f"⏰ {req['timestamp']}")
                    st.caption(f"🔄 Status: {req['status']}")
                    st.caption(f"⚡ Time: {req['duration']:.2f}s")
                    if req.get('error'):
                        st.error(f"❌ {req['error']}")
                    st.write("---")
        else:
            st.info("Chưa có requests nào")
        
        if st.button("🗑️ Xóa lịch sử", key="clear_api_history"):
            st.session_state.api_requests = []
            st.rerun()

def log_api_request(status, duration, error=None, prompt_details=None):
    """Log API request vào session state với thông tin prompt"""
    if 'api_requests' not in st.session_state:
        st.session_state.api_requests = []
    
    request_log = {
        'timestamp': time.strftime('%H:%M:%S'),
        'status': status,
        'duration': duration,
        'error': error,
        'prompt_details': prompt_details or {}
    }
    
    st.session_state.api_requests.append(request_log)
    
    if len(st.session_state.api_requests) > 20:
        st.session_state.api_requests = st.session_state.api_requests[-20:]

def serialize_mcp_result(obj):
    """Chuyển đổi TextContent hoặc các đối tượng không JSON-serializable thành chuỗi"""
    if hasattr(obj, 'text'):
        return str(obj.text)
    elif hasattr(obj, '__str__'):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def display_prompt_details(question, context, model_config, response=None, duration=None):
    """Hiển thị chi tiết prompt được gửi lên model"""
    with st.expander("🔍 Chi tiết Prompt & Request", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["📤 Request", "📝 Prompt", "⚙️ Config", "📥 Response"])
        
        with tab1:
            st.subheader("Request Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Model Settings:**")
                config = load_config_local()
                st.write(f"- **Base URL:** `{config['LOCAL_MODEL']['base_url']}`")
                st.write(f"- **Model:** `{model_config['selected_model']}`")
                st.write(f"- **Max Tokens:** {model_config['max_tokens']}")
                st.write(f"- **Temperature:** {model_config['temperature']}")
                st.write(f"- **Top P:** {model_config['top_p']}")
                st.write(f"- **Timeout:** {model_config['timeout']}s")
                st.write(f"- **Streaming:** {model_config.get('enable_streaming', False)}")
            
            with col2:
                st.write("**📊 Request Stats:**")
                if duration:
                    st.write(f"- **Duration:** {duration:.2f}s")
                st.write(f"- **Question Length:** {len(question)} chars")
                st.write(f"- **Context Length:** {len(context)} chars")
                st.write(f"- **Total Prompt Length:** {len(context) + len(question)} chars")
                if response:
                    st.write(f"- **Response Length:** {len(response)} chars")
                    st.write(f"- **Response Words:** {len(response.split())}")
        
        with tab2:
            st.subheader("Full Prompt Sent to Model")
            full_prompt = create_intelligent_multimodal_context(
                context.replace(model_config['system_prompt'], '').strip(),
                [],
                question,
                model_config['system_prompt']
            )
            
            st.write("**🤖 System Prompt:**")
            st.code(model_config['system_prompt'], language="text")
            
            st.write("**❓ User Question:**")
            st.code(question, language="text")
            
            st.write("**📚 Context Provided:**")
            context_only = context.replace(model_config['system_prompt'], '').strip()
            if context_only:
                if len(context_only) > 2000:
                    st.code(context_only[:2000] + "\n\n... (truncated)", language="text")
                    st.caption(f"Full context: {len(context_only)} characters")
                else:
                    st.code(context_only, language="text")
            else:
                st.info("No additional context provided")
            
            st.write("**🔗 Complete Prompt:**")
            if len(full_prompt) > 3000:
                st.code(full_prompt[:3000] + "\n\n... (truncated for display)", language="text")
                st.caption(f"Full prompt: {len(full_prompt)} characters")
            else:
                st.code(full_prompt, language="text")
        
        with tab3:
            st.subheader("Model Configuration")
            config_dict = {
                "model": model_config['selected_model'],
                "max_tokens": model_config['max_tokens'],
                "temperature": model_config['temperature'],
                "top_p": model_config['top_p'],
                "timeout": model_config['timeout'],
                "stream": model_config.get('enable_streaming', False)
            }
            
            st.write("**JSON Configuration:**")
            st.code(json.dumps(config_dict, indent=2), language="json")
            
            st.write("**Equivalent API Call:**")
            config = load_config_local()
            api_call = f"""curl -X POST "{config['LOCAL_MODEL']['base_url']}/api/generate" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model_config['selected_model']}",
    "prompt": "{{PROMPT_TEXT}}",
    "stream": {str(model_config.get('enable_streaming', False)).lower()},
    "options": {{
      "temperature": {model_config['temperature']},
      "top_p": {model_config['top_p']},
      "num_predict": {model_config['max_tokens']}
    }}
  }}'"""
            st.code(api_call, language="bash")
        
        with tab4:
            if response:
                st.subheader("Model Response")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📈 Response Metrics:**")
                    st.write(f"- **Length:** {len(response)} characters")
                    st.write(f"- **Words:** {len(response.split())}")
                    st.write(f"- **Lines:** {len(response.split('\n'))}")
                    if duration:
                        words_per_sec = len(response.split()) / duration if duration > 0 else 0
                        st.write(f"- **Speed:** {words_per_sec:.1f} words/sec")
                
                with col2:
                    st.write("**🎯 Response Quality:**")
                    has_vietnamese = bool(re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', response))
                    has_structure = bool(re.search(r'(\n\n|\d+\.|•|-)', response))
                    has_images = bool(re.search(r'\[IMAGE_\d+\]', response))
                    
                    st.write(f"- **Vietnamese:** {'✅' if has_vietnamese else '❌'}")
                    st.write(f"- **Structured:** {'✅' if has_structure else '❌'}")
                    st.write(f"- **Has Images:** {'✅' if has_images else '❌'}")
                
                st.write("**📝 Full Response:**")
                st.code(response, language="text")
            else:
                st.info("No response available yet")

def create_sidebar():
    """Tạo sidebar với các chức năng điều khiển"""
    st.sidebar.title("🔧 Điều khiển hệ thống")
    
    st.sidebar.header("🤖 Chọn Model")
    
    available_models = get_available_models()
    
    config = load_config_local()
    default_model = config.get('LOCAL_MODEL', {}).get('model', 'qwen2.5:3b')
    
    if default_model not in available_models:
        available_models.insert(0, default_model)
    
    selected_model = st.sidebar.selectbox(
        "Chọn Model:",
        options=available_models,
        index=0,
        help="Chọn model AI để sử dụng cho việc trả lời câu hỏi",
        key="model_selector"
    )
    
    if selected_model:
        try:
            local_client = LocalGemmaClient(
                base_url=config['LOCAL_MODEL']['base_url'],
                model=selected_model
            )
            model_exists = local_client.check_model_exists(selected_model)
            
            if model_exists:
                st.sidebar.success(f"✅ Model {selected_model} sẵn sàng")
            else:
                st.sidebar.error(f"❌ Model {selected_model} không tồn tại")
                
                if st.sidebar.button(f"📥 Tải model {selected_model}", key="download_model"):
                    with st.sidebar.spinner(f"Đang tải {selected_model}..."):
                        success = local_client.pull_model(selected_model)
                        if success:
                            st.sidebar.success(f"✅ Đã tải {selected_model} thành công!")
                            st.rerun()
                        else:
                            st.sidebar.error(f"❌ Không thể tải {selected_model}")
                            
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi kiểm tra model: {e}")
    
    display_server_info(selected_model)
    
    display_api_monitor()
    
    with st.sidebar.expander("⚙️ Cấu hình Model", expanded=False):
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=100,
            max_value=4000,
            value=config.get('LOCAL_MODEL', {}).get('max_tokens', 2000),
            step=100,
            help="Số token tối đa cho response",
            key="max_tokens_slider"
        )
        
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=config.get('LOCAL_MODEL', {}).get('temperature', 0.3),
            step=0.1,
            help="Độ sáng tạo của model (0 = deterministic, 2 = very creative)",
            key="temperature_slider"
        )
        
        top_p = st.slider(
            "Top P:",
            min_value=0.1,
            max_value=1.0,
            value=config.get('LOCAL_MODEL', {}).get('top_p', 0.4),
            step=0.1,
            help="Nucleus sampling parameter",
            key="top_p_slider"
        )
        
        timeout = st.slider(
            "Timeout (seconds):",
            min_value=30,
            max_value=300,
            value=config.get('LOCAL_MODEL', {}).get('timeout', 180),
            step=30,
            help="Thời gian chờ tối đa cho response",
            key="timeout_slider"
        )
        
        enable_streaming = st.checkbox(
            "🌊 Bật Streaming",
            value=True,
            help="Hiển thị response theo thời gian thực"
        )
        
        show_prompt_details = st.checkbox(
            "🔍 Hiển thị chi tiết Prompt",
            value=True,
            help="Hiển thị prompt đầy đủ được gửi lên model"
        )
    
    with st.sidebar.expander("📝 System Prompt", expanded=False):
        default_system_prompt = """"""
        
        system_prompt = st.text_area(
            "Custom System Prompt:",
            value=default_system_prompt,
            height=200,
            help="Tùy chỉnh system prompt cho model",
            key="system_prompt_textarea"
        )
        
        st.write("**Prompt Templates:**")
        if st.button("📚 Academic", key="academic_prompt"):
            st.session_state.system_prompt_textarea = """Bạn là một trợ lý học thuật chuyên nghiệp. Hãy trả lời với phong cách học thuật, có trích dẫn và phân tích sâu."""
            st.rerun()
        
        if st.button("💼 Business", key="business_prompt"):
            st.session_state.system_prompt_textarea = """Bạn là một cố vấn kinh doanh. Hãy trả lời với góc nhìn thực tế, tập trung vào giải pháp và hiệu quả."""
            st.rerun()
        
        if st.button("🎓 Tutorial", key="tutorial_prompt"):
            st.session_state.system_prompt_textarea = """Bạn là một giảng viên. Hãy giải thích từng bước một cách dễ hiểu, có ví dụ cụ thể."""
            st.rerun()
    
    with st.sidebar.expander("🔍 Cấu hình Tìm kiếm", expanded=False):
        search_threshold = st.slider(
            "Ngưỡng tìm kiếm:",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.05,
            help="Ngưỡng độ tương tự tối thiểu"
        )
        
        max_results = st.selectbox(
            "Số kết quả tối đa:",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=2,
            help="Số documents tối đa để tìm kiếm"
        )
        
        max_images = st.slider(
            "Số ảnh tối đa:",
            min_value=1,
            max_value=10,
            value=5,
            help="Số ảnh tối đa hiển thị"
        )
        
        enable_fallback = st.checkbox(
            "🔄 Bật AI Fallback",
            value=True,
            help="Sử dụng AI khi không tìm thấy kết quả"
        )
    
    with st.sidebar.expander("📥 Thu thập dữ liệu", expanded=False):
        st.write("**Thu thập từ URL**")
        new_url = st.text_input(
            "Nhập URL cần thu thập:",
            placeholder="https://example.com/article",
            key="new_url_input"
        )
        
        st.write("**Nhập nội dung văn bản**")
        new_text = st.text_area(
            "Nhập nội dung để thêm vào database:",
            placeholder="Nhập văn bản tại đây...",
            height=150,
            key="new_text_input"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Thu thập URL", key="collect_single", type="primary"):
                if new_url:
                    collect_single_url(new_url)
                else:
                    st.error("Vui lòng nhập URL")
        
        with col2:
            if st.button("💾 Lưu nội dung", key="save_text"):
                if new_text.strip():
                    text_collector = TextContentCollector()
                    success = text_collector.save_text_content(new_text)
                    if success:
                        st.success("✅ Đã lưu nội dung vào database")
                        if 'rag_system' in st.session_state:
                            st.session_state.rag_system.load_multimodal_data()
                        st.rerun()
                    else:
                        st.error("❌ Không thể lưu nội dung")
                else:
                    st.error("Vui lòng nhập nội dung")
        
        with col3:
            if st.button("🗑️ Xóa DB", key="clear_db"):
                clear_database()
    
    with st.sidebar.expander("🌐 Cấu hình MCP Server", expanded=False):
        mcp_config = config.get('MCP', {})
        
        mcp_sse_url = st.text_input(
            "MCP SSE URL:",
            value=mcp_config.get('sse_url', 'http://localhost:8081/sse'),
            key="mcp_sse_url_input"
        )
        
        mcp_timeout = st.slider(
            "Timeout (seconds):",
            min_value=10,
            max_value=120,
            value=mcp_config.get('timeout', 30),
            step=10,
            key="mcp_timeout_slider"
        )
        
        if st.button("🔍 Kiểm tra kết nối MCP", key="test_mcp_connection"):
            with st.spinner("Đang kiểm tra kết nối..."):
                mcp_client = MCPClient(mcp_sse_url, mcp_timeout)
                try:
                    success, message = asyncio.run(mcp_client.test_connection())
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.mcp_client = mcp_client
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối: {str(e)}")
                finally:
                    if 'mcp_client' in locals() and mcp_client != st.session_state.get('mcp_client'):
                        mcp_client.close()

    return {
        'selected_model': selected_model,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'timeout': timeout,
        'system_prompt': system_prompt,
        'enable_streaming': enable_streaming,
        'search_threshold': search_threshold,
        'max_results': max_results,
        'max_images': max_images,
        'enable_fallback': enable_fallback,
        'show_prompt_details': show_prompt_details,
        'mcp_config': {
            'sse_url': mcp_sse_url,
            'timeout': mcp_timeout
        }
    }

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

def clear_database():
    """Xóa database"""
    try:
        db_path = Path("db")
        if db_path.exists():
            shutil.rmtree(db_path)
            st.success("✅ Đã xóa database")
            if 'rag_system' in st.session_state:
                st.session_state.rag_system.load_multimodal_data()
            st.rerun()
        else:
            st.warning("Database không tồn tại")
    except Exception as e:
        st.error(f"Lỗi xóa database: {e}")

def handle_no_results_fallback(question, model_config):
    """Xử lý fallback khi không tìm thấy kết quả"""
    if not model_config.get('enable_fallback', True):
        return "Không tìm thấy thông tin liên quan và AI fallback đã bị tắt."
    
    st.info("🔍 Không tìm thấy thông tin liên quan trong database hoặc MCP. Đang gọi AI để trả lời...")
    
    fallback_context = f"""
{model_config['system_prompt']}

{question}
"""
    
    try:
        config = load_config_local()
        temp_config = config.copy()
        temp_config['LOCAL_MODEL'].update({
            'model': model_config['selected_model'],
            'max_tokens': model_config['max_tokens'],
            'temperature': model_config['temperature'],
            'top_p': model_config['top_p'],
            'timeout': model_config['timeout']
        })
        
        start_time = time.time()
        
        if model_config.get('enable_streaming', False):
            response = ask_local_model_streaming(question, fallback_context, temp_config)
        else:
            response = ask_local_model(question, fallback_context, temp_config)
        
        duration = time.time() - start_time
        
        prompt_details = {
            'type': 'fallback',
            'question': question,
            'context_length': len(fallback_context),
            'system_prompt_length': len(model_config['system_prompt']),
            'model': model_config['selected_model']
        }
        
        log_api_request("Success (Fallback)", duration, prompt_details=prompt_details)
        
        if model_config.get('show_prompt_details', True):
            display_prompt_details(question, fallback_context, model_config, response, duration)
        
        st.info("💡 **Lưu ý:** Câu trả lời dựa trên kiến thức chung của AI.")
        return response
        
    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        log_api_request("Error", duration, str(e))
        logger.error(f"Error in fallback AI response: {e}")
        return f"Xin lỗi, không thể trả lời do lỗi: {e}"

def ask_local_model_streaming(question, context, config):
    """Hỏi model local với streaming"""
    try:
        local_client = LocalGemmaClient(
            base_url=config["LOCAL_MODEL"]["base_url"],
            model=config["LOCAL_MODEL"]["model"]
        )
        
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
        raise e

def create_intelligent_multimodal_context(text_context, relevant_images, question, system_prompt):
    """Tạo context đa phương tiện thông minh"""
    context = f"{system_prompt}\n\nThông tin văn bản:\n{text_context}\n\n"
    
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

Câu hỏi: {question}

Trả lời:
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

def smart_display_answer_with_embedded_images(answer, relevant_images, enable_streaming=False):
    """Hiển thị câu trả lời với ảnh chèn thông minh"""
    if not relevant_images:
        if enable_streaming:
            display_streaming_text(answer)
        else:
            st.markdown(answer)
        return
    
    content_parts = parse_answer_with_image_markers(answer, relevant_images)
    
    if not any(part['type'] == 'image' for part in content_parts):
        auto_embed_images_in_answer(answer, relevant_images, enable_streaming)
        return
    
    for part in content_parts:
        if part['type'] == 'text':
            if enable_streaming:
                display_streaming_text(part['content'])
            else:
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

def display_streaming_text(text):
    """Hiển thị text với hiệu ứng streaming"""
    placeholder = st.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(0.01)

def auto_embed_images_in_answer(answer, relevant_images, enable_streaming=False):
    """Tự động chèn ảnh vào câu trả lời"""
    paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
    total_images = len(relevant_images)
    
    if total_images == 0:
        if enable_streaming:
            display_streaming_text(answer)
        else:
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
        if enable_streaming:
            display_streaming_text(paragraph)
        else:
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
        page_title="RAG Local với Ảnh Chèn & MCP SSE",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    model_config = create_sidebar()
    
    st.title("🤖 DEMO AI")
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
            st.write("1. Sử dụng sidebar để thêm URL hoặc nội dung văn bản")
            st.write("2. Cấu hình MCP server qua SSE")
            st.write("3. Nhấn 'Thu thập' hoặc 'Lưu nội dung' để tải dữ liệu")
            st.write("4. Đặt câu hỏi sau khi có dữ liệu")
        
        with col2:
            st.info("🔧 **Tính năng:**")
            st.write("- Thu thập từ URL")
            st.write("- Thêm nội dung văn bản trực tiếp")
            st.write("- Ảnh chèn thông minh")
            st.write("- AI trả lời mọi câu hỏi")
            st.write("- Cấu hình model linh hoạt")
            st.write("- Streaming response")
            st.write("- API monitoring")
            st.write("- Chi tiết Prompt debugging")
            st.write("- Chọn model từ danh sách")
            st.write("- Kết nối MCP qua SSE")
    
    if rag_system.has_data:
        display_database_debug_info(rag_system)
    
    st.markdown("---")
    
    st.header("💬 Hỏi đáp thông minh với AI & MCP")
    
    st.info(f"🤖 **Model hiện tại:** {model_config['selected_model']}")
    
    question = st.text_input(
        "Đặt câu hỏi:",
        placeholder="Bạn có thể hỏi bất kỳ điều gì...",
        help="Nhập câu hỏi và nhấn Enter."
    )
    
    if question:
        config = load_config_local()
        is_valid, validation_message = validate_query_input(question, config)
        
        if not is_valid:
            st.error(f"❌ {validation_message}")
            return
        
        with st.spinner("🔍 Đang tìm kiếm và tạo câu trả lời..."):
            try:
                start_time = time.time()
                
                enhanced_query = question
                mcp_result = None
                mcp_context = ""
                if 'mcp_client' in st.session_state:
                    mcp_client = st.session_state.mcp_client
                    try:
                        # Lấy prompt cho AI chọn tool
                        tool_prompt, _ = asyncio.run(mcp_client.process_query(question))
                        logger.info(f"MCP tool prompt: {tool_prompt}")
                        if tool_prompt and not tool_prompt.startswith("Lỗi"):
                            # Gửi prompt đến AI để chọn tool
                            temp_config = config.copy()
                            temp_config['LOCAL_MODEL'].update({
                                'model': model_config['selected_model'],
                                'max_tokens': model_config['max_tokens'],
                                'temperature': model_config['temperature'],
                                'top_p': model_config['top_p'],
                                'timeout': model_config['timeout']
                            })
                            tool_selection = ask_local_model(tool_prompt, "", temp_config)
                            logger.info(f"AI tool selection: {tool_selection}")
                            
                            try:
                                tool_selection = tool_selection.replace('```json', '')
                                tool_selection = tool_selection.replace('```', '')
                                tool_info = json.loads(tool_selection)
                                if tool_info and "tool_name" in tool_info:
                                    # Gọi tool được AI chọn
                                    tool_name = tool_info["tool_name"]
                                    tool_params = tool_info.get("parameters", {})
                                    mcp_result = asyncio.run(mcp_client.call_tool(tool_name, tool_params))
                                    if mcp_result:
                                        mcp_context = f"Kết quả từ công cụ {tool_name}: {json.dumps(mcp_result, ensure_ascii=False, default=serialize_mcp_result)}"
                                        st.info(f"🔧 **Kết quả MCP Tool {tool_name}:** {mcp_result}")
                                    else:
                                        st.warning(f"⚠️ Công cụ {tool_name} {tool_selection} không trả về kết quả")
                                
                            except json.JSONDecodeError:
                                logger.error(f"AI trả về định dạng JSON không hợp lệ: {tool_selection}")
                                st.warning(f"⚠️ AI trả về lựa chọn công cụ không hợp lệ: {tool_selection}")
                                # Fallback cho câu hỏi về bug
                                if "bug" in question.lower():
                                    tool_name = "totalBug"
                                    target = question.split()[0].capitalize() if question.split() else "Unknown"
                                    tool_params = {"member_name": target}
                                    mcp_result = asyncio.run(mcp_client.call_tool(tool_name, tool_params))
                                    if mcp_result:
                                        mcp_context = f"Kết quả từ công cụ {tool_name}: {json.dumps(mcp_result, ensure_ascii=False, default=serialize_mcp_result)}"
                                        st.info(f"🔧 **Kết quả MCP Tool {tool_name} (fallback):** {mcp_result}")
                                    else:
                                        st.warning(f"⚠️ Fallback công cụ {tool_name} không trả về kết quả")
                        else:
                            st.error(f"❌ Không thể tạo prompt cho MCP tool: {tool_prompt if tool_prompt else 'Không có prompt'}")
                    except Exception as e:
                        logger.warning(f"MCP tool processing failed: {e}")
                        st.error(f"❌ Không thể xử lý MCP tool: {e}")
                        # Fallback cho câu hỏi về bug
                        if "bug" in question.lower() and 'mcp_client' in st.session_state:
                            tool_name = "totalBug"
                            target = question.split()[0].capitalize() if question.split() else "Unknown"
                            tool_params = {"member_name": target}
                            mcp_result = asyncio.run(mcp_client.call_tool(tool_name, tool_params))
                            if mcp_result:
                                mcp_context = f"Kết quả từ công cụ {tool_name}: {json.dumps(mcp_result, ensure_ascii=False, default=serialize_mcp_result)}"
                                st.info(f"🔧 **Kết quả MCP Tool {tool_name} (fallback):** {mcp_result}")
                            else:
                                st.warning(f"⚠️ Fallback công cụ {tool_name} không trả về kết quả")
                
                # Tìm kiếm trong database với enhanced_query
                relevant_docs, similarity = rag_system.enhanced_search_multimodal(
                    enhanced_query, model_config['search_threshold'], model_config['max_results']
                )
                
                st.subheader("🔍 Kết quả tìm kiếm")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Độ tương tự tối đa", f"{similarity:.3f}")
                with col2:
                    st.metric("Số documents", len(relevant_docs))
                with col3:
                    st.metric("Ngưỡng sử dụng", f"{model_config['search_threshold']:.3f}")
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
                            for word in enhanced_query.split():
                                if len(word) > 3:
                                    highlighted_snippet = highlighted_snippet.replace(
                                        word, f"**{word}**"
                                    )
                            
                            st.write(f"*Đoạn trích:* {highlighted_snippet}")
                            st.write("---")
                
                # Tích hợp kết quả MCP và database vào context
                text_context = mcp_context
                if relevant_docs:
                    text_context += "\n\n" + "\n\n".join([
                        f"Tiêu đề: {doc['title']}\nMô tả: {doc['description']}\nNội dung: {doc['text_content'][:1000]}..."
                        for doc in relevant_docs
                    ])
                
                relevant_images = rag_system.get_relevant_images_for_context(
                    relevant_docs, enhanced_query, model_config['max_images']
                )
                
                context = create_intelligent_multimodal_context(
                    text_context, relevant_images, question, model_config['system_prompt']
                )
                
                temp_config = config.copy()
                temp_config['LOCAL_MODEL'].update({
                    'model': model_config['selected_model'],
                    'max_tokens': model_config['max_tokens'],
                    'temperature': model_config['temperature'],
                    'top_p': model_config['top_p'],
                    'timeout': model_config['timeout']
                })
                
                if model_config.get('enable_streaming', False):
                    answer = ask_local_model_streaming(question, context, temp_config)
                else:
                    answer = ask_local_model(question, context, temp_config)
                
                duration = time.time() - start_time
                
                prompt_details = {
                    'type': 'database_search_with_mcp',
                    'question': question,
                    'context_length': len(context),
                    'system_prompt_length': len(model_config['system_prompt']),
                    'documents_found': len(relevant_docs),
                    'images_found': len(relevant_images),
                    'mcp_result': str(mcp_result) if mcp_result else None,
                    'model': model_config['selected_model']
                }
                
                log_api_request("Success", duration, prompt_details=prompt_details)
                
                st.subheader("📝 Câu trả lời:")
                st.success("✅ **Dựa trên database và MCP**")
                
                smart_display_answer_with_embedded_images(
                    answer, relevant_images, model_config.get('enable_streaming', False)
                )
                
                if model_config.get('show_prompt_details', True):
                    display_prompt_details(question, context, model_config, answer, duration)
                
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
            
            except Exception as e:
                duration = time.time() - start_time if 'start_time' in locals() else 0
                log_api_request("Error", duration, str(e))
                st.error(f"❌ Lỗi khi xử lý câu hỏi: {e}")
                logger.error(f"Lỗi: {e}")
                
                st.info("🔄 Đang thử phương pháp dự phòng...")
                try:
                    emergency_answer = handle_no_results_fallback(question, model_config)
                    st.subheader("📝 Câu trả lời (Dự phòng):")
                    
                    if model_config.get('enable_streaming', False):
                        display_streaming_text(emergency_answer)
                    else:
                        st.markdown(emergency_answer)
                        
                except Exception as e2:
                    st.error(f"❌ Không thể trả lời: {e2}")
    
    st.markdown("---")
    st.markdown("🤖 **RAG Local System với MCP SSE** - Trả lời dựa trên database, MCP, hoặc kiến thức chung")

if __name__ == "__main__":
    main()