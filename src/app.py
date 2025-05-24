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
from config import load_config
from rag import load_documents, search_documents, ask_gemini, search_documents_with_threshold

class MultimodalRAG:
    def __init__(self, db_dir):
        self.db_dir = Path(db_dir)
        self.documents = []
        self.has_data = False
        self.load_multimodal_data()
    
    def load_multimodal_data(self):
        """Load dữ liệu đa phương tiện từ database"""
        self.documents = []
        
        print(f"🔍 Kiểm tra thư mục db: {self.db_dir.absolute()}")
        
        if not self.db_dir.exists():
            print(f"❌ Thư mục db không tồn tại: {self.db_dir}")
            
            parent_dir = self.db_dir.parent
            print(f"📁 Thư mục cha: {parent_dir}")
            if parent_dir.exists():
                print(f"📋 Nội dung thư mục cha:")
                for item in parent_dir.iterdir():
                    item_type = "📁" if item.is_dir() else "📄"
                    print(f"  {item_type} {item.name}")
            
            self.has_data = False
            return
            
        site_dirs = [d for d in self.db_dir.iterdir() if d.is_dir()]
        print(f"📁 Tìm thấy {len(site_dirs)} thư mục con trong db")
        
        for site_dir in site_dirs:
            print(f"🔍 Kiểm tra thư mục: {site_dir.name}")
            
            metadata_file = site_dir / "metadata.json"
            if not metadata_file.exists():
                print(f"❌ Thiếu metadata.json trong {site_dir.name}")
                continue
                
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"✅ Đọc metadata thành công từ {site_dir.name}")
                
                content_file = site_dir / "content.txt"
                text_content = ""
                if content_file.exists():
                    with open(content_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    print(f"✅ Đọc content.txt thành công ({len(text_content)} ký tự)")
                else:
                    print(f"⚠️ Không tìm thấy content.txt trong {site_dir.name}")
                
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
                print(f"✅ Thêm document: {doc['title']}")
                
            except Exception as e:
                print(f"❌ Lỗi load dữ liệu từ {site_dir}: {e}")
                st.warning(f"Lỗi load dữ liệu từ {site_dir}: {e}")
        
        self.has_data = len(self.documents) > 0
        print(f"📊 Tổng cộng: {len(self.documents)} documents")
    
    def search_multimodal(self, query, threshold=0.3, top_k=3):
        """Tìm kiếm đa phương tiện với threshold"""
        if not self.has_data:
            return [], 0
            
        text_corpus = []
        for doc in self.documents:
            combined_text = f"{doc['title']}\n{doc['description']}\n{doc['text_content']}"
            
            for img in doc['images']:
                if img.get('alt'):
                    combined_text += f"\n{img['alt']}"
                if img.get('title'):
                    combined_text += f"\n{img['title']}"
            
            text_corpus.append(combined_text)
        
        try:
            relevant_texts, max_similarity = search_documents_with_threshold(
                query, text_corpus, threshold, top_k
            )
            
            relevant_docs = []
            for relevant_text in relevant_texts:
                for i, text in enumerate(text_corpus):
                    if text == relevant_text:
                        relevant_docs.append(self.documents[i])
                        break
            
            return relevant_docs, max_similarity
        except Exception as e:
            print(f"❌ Lỗi search: {e}")
            return [], 0
    
    def get_relevant_images_for_context(self, relevant_docs, query, max_images=5):
        """Lấy ảnh liên quan để đưa vào context cho Gemini"""
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

def create_intelligent_multimodal_context(text_context, relevant_images, question):
    """Tạo context đa phương tiện thông minh cho Gemini"""
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

def parse_answer_with_image_markers(answer, relevant_images):
    """Phân tích câu trả lời và tách các marker ảnh"""
    # Tìm tất cả các marker [IMAGE_X]
    image_pattern = r'\[IMAGE_(\d+)\]'
    markers = re.findall(image_pattern, answer)
    
    # Tách câu trả lời thành các phần
    parts = re.split(image_pattern, answer)
    
    # Tạo danh sách các phần với thông tin về ảnh
    content_parts = []
    part_index = 0
    
    for i, part in enumerate(parts):
        if part.isdigit():  # Đây là số của marker ảnh
            image_index = int(part) - 1  # Chuyển từ 1-based sang 0-based
            if 0 <= image_index < len(relevant_images):
                content_parts.append({
                    'type': 'image',
                    'content': relevant_images[image_index],
                    'index': image_index
                })
        else:  # Đây là text
            if part.strip():  # Chỉ thêm nếu không phải chuỗi rỗng
                content_parts.append({
                    'type': 'text',
                    'content': part.strip()
                })
    
    return content_parts

def smart_display_answer_with_embedded_images(answer, relevant_images):
    """Hiển thị câu trả lời với ảnh được chèn thông minh dựa trên AI"""
    
    if not relevant_images:
        st.markdown(answer)
        return
    
    # Phân tích câu trả lời để tìm các marker ảnh
    content_parts = parse_answer_with_image_markers(answer, relevant_images)
    
    if not any(part['type'] == 'image' for part in content_parts):
        # Nếu AI không sử dụng marker, fallback về phương pháp tự động
        auto_embed_images_in_answer(answer, relevant_images)
        return
    
    # Hiển thị từng phần
    for part in content_parts:
        if part['type'] == 'text':
            st.markdown(part['content'])
        elif part['type'] == 'image':
            img_info = part['content']
            try:
                image = Image.open(img_info['path'])
                
                # Tạo caption thông minh
                caption = ""
                if img_info.get('alt'):
                    caption = f"📷 {img_info['alt']}"
                elif img_info.get('title'):
                    caption = f"📷 {img_info['title']}"
                else:
                    caption = f"📷 Hình ảnh từ {img_info['source_doc']['title']}"
                
                # Hiển thị ảnh với styling đẹp
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                
                # Thêm khoảng cách
                st.markdown("<br>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")

def auto_embed_images_in_answer(answer, relevant_images):
    """Tự động chèn ảnh vào câu trả lời khi AI không sử dụng marker"""
    
    # Tách câu trả lời thành các đoạn
    paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
    
    if len(paragraphs) <= 1:
        # Nếu chỉ có 1 đoạn, tách theo câu
        sentences = [s.strip() + '.' for s in answer.split('.') if s.strip()]
        paragraphs = sentences
    
    # Tính toán vị trí chèn ảnh
    total_parts = len(paragraphs)
    total_images = len(relevant_images)
    
    if total_images == 0:
        st.markdown(answer)
        return
    
    # Tạo danh sách vị trí chèn ảnh
    insert_positions = []
    if total_parts > 1:
        # Phân bố đều ảnh trong câu trả lời
        step = max(1, total_parts // (total_images + 1))
        for i in range(min(total_images, total_parts - 1)):
            pos = (i + 1) * step
            if pos < total_parts:
                insert_positions.append(pos)
    
    # Hiển thị với ảnh được chèn
    image_index = 0
    for i, paragraph in enumerate(paragraphs):
        # Hiển thị đoạn văn
        st.markdown(paragraph)
        
        # Chèn ảnh nếu đến vị trí phù hợp
        if i in insert_positions and image_index < len(relevant_images):
            img_info = relevant_images[image_index]
            
            try:
                image = Image.open(img_info['path'])
                
                # Tạo caption
                caption = ""
                if img_info.get('alt'):
                    caption = f"📷 {img_info['alt']}"
                elif img_info.get('title'):
                    caption = f"📷 {img_info['title']}"
                else:
                    caption = f"📷 Hình ảnh minh họa"
                
                # Hiển thị ảnh với layout đẹp
                st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption=caption, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                image_index += 1
                
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")
    
    # Hiển thị ảnh còn lại ở cuối
    if image_index < len(relevant_images):
        st.markdown("### 🖼️ Hình ảnh bổ sung")
        remaining_images = relevant_images[image_index:]
        
        cols = st.columns(min(len(remaining_images), 3))
        for i, img_info in enumerate(remaining_images):
            with cols[i % len(cols)]:
                try:
                    image = Image.open(img_info['path'])
                    st.image(image, use_container_width=True)
                    if img_info.get('alt') or img_info.get('title'):
                        st.caption(img_info.get('alt') or img_info.get('title'))
                except Exception as e:
                    st.error(f"Không thể hiển thị ảnh: {e}")

def ask_gemini_with_intelligent_images(question, text_context, relevant_images, api_key, model_name):
    """Hỏi Gemini với context thông minh bao gồm hướng dẫn chèn ảnh"""
    try:
        # Tạo context với hướng dẫn chèn ảnh thông minh
        intelligent_context = create_intelligent_multimodal_context(text_context, relevant_images, question)
        
        # Gọi Gemini với context đã được tối ưu
        response = ask_gemini(question, intelligent_context, api_key, model_name)
        
        return response, relevant_images
    except Exception as e:
        return f"Xin lỗi, tôi không thể trả lời câu hỏi này. Lỗi: {e}", []

def ask_gemini_direct(question, api_key, model_name):
    """Hỏi Gemini trực tiếp không cần context"""
    try:
        return ask_gemini(question, "", api_key, model_name)
    except Exception as e:
        return f"Xin lỗi, tôi không thể trả lời câu hỏi này. Lỗi: {e}"

def enhanced_search_with_fallback(rag_system, question, threshold):
    """Tìm kiếm nâng cao với fallback thông minh"""
    try:
        relevant_docs, max_similarity = rag_system.search_multimodal(question, threshold)
        
        if not relevant_docs:
            return "no_results", None, None, None, 0
        
        if max_similarity < threshold:
            return "low_relevance", None, None, None, max_similarity
        
        text_context = "\n---\n".join([
            f"Tiêu đề: {doc['title']}\nMô tả: {doc['description']}\nNội dung: {doc['text_content'][:1500]}..."
            for doc in relevant_docs
        ])
        
        relevant_images = rag_system.get_relevant_images_for_context(
            relevant_docs, question, 5  # Tăng số ảnh để có nhiều lựa chọn
        )
        
        return "success", relevant_docs, text_context, relevant_images, max_similarity
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình tìm kiếm: {e}")
        return "error", None, None, None, 0

def check_database_status(db_dir):
    """Kiểm tra trạng thái database và hiển thị thông tin debug"""
    db_path = Path(db_dir)
    
    with st.expander("🔧 Thông tin Debug", expanded=False):
        st.code(f"""
Thông tin đường dẫn:
- File app.py: {Path(__file__).absolute()}
- Thư mục src: {Path(__file__).parent.absolute()}
- Thư mục project: {Path(__file__).parent.parent.absolute()}
- Thư mục db: {db_path.absolute()}
- DB exists: {db_path.exists()}
        """)
        
        if db_path.exists():
            subdirs = [d for d in db_path.iterdir() if d.is_dir()]
            st.write(f"📁 Số thư mục con: {len(subdirs)}")
            
            for subdir in subdirs:
                metadata_file = subdir / "metadata.json"
                content_file = subdir / "content.txt"
                images_dir = subdir / "images"
                
                st.write(f"📂 {subdir.name}:")
                st.write(f"  - metadata.json: {'✅' if metadata_file.exists() else '❌'}")
                st.write(f"  - content.txt: {'✅' if content_file.exists() else '❌'}")
                st.write(f"  - images/: {'✅' if images_dir.exists() else '❌'}")

def display_search_status(search_result, relevance_score=None, threshold=None):
    """Hiển thị trạng thái tìm kiếm cho user"""
    if search_result == "success":
        st.success(f"✅ Tìm thấy thông tin trong database (điểm: {relevance_score:.3f})")
    elif search_result == "no_results":
        st.info("🔍 Không tìm thấy thông tin trong database → Sử dụng AI")
    elif search_result == "low_relevance":
        st.warning(f"📊 Độ liên quan thấp ({relevance_score:.3f} < {threshold}) → Sử dụng AI")
    elif search_result == "error":
        st.error("⚠️ Lỗi tìm kiếm → Sử dụng AI")

def main():
    st.set_page_config(
        page_title="Hệ thống RAG Đa phương tiện với AI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Hệ thống hỏi đáp RAG đa phương tiện với AI")
    st.markdown("*Chèn ảnh thông minh bằng AI vào câu trả lời*")
    st.markdown("---")
    
    try:
        config = load_config()
        
        # Xác định đường dẫn db
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        db_dir = project_root / "db"
        
        check_database_status(db_dir)
        
        # Initialize multimodal RAG
        if 'multimodal_rag' not in st.session_state:
            with st.spinner("Đang kiểm tra dữ liệu..."):
                st.session_state.multimodal_rag = MultimodalRAG(str(db_dir))
        
        rag_system = st.session_state.multimodal_rag
        
        # Sidebar
        with st.sidebar:
            st.header("📊 Thông tin Hệ thống")
            
            if rag_system.has_data:
                st.success("✅ Database có dữ liệu")
                st.write(f"**Số trang web:** {len(rag_system.documents)}")
                
                total_images = sum(len(doc['images']) for doc in rag_system.documents)
                st.write(f"**Tổng số ảnh:** {total_images}")
                
                st.markdown("**Các trang web:**")
                for doc in rag_system.documents:
                    with st.expander(f"🌐 {doc['title'][:50]}..."):
                        st.write(f"**URL:** {doc['url']}")
                        st.write(f"**Số ảnh:** {len(doc['images'])}")
                        st.write(f"**Mô tả:** {doc['description'][:100]}...")
                        
                # Settings
                st.markdown("### ⚙️ Cài đặt RAG")
                relevance_threshold = st.slider(
                    "Ngưỡng độ liên quan", 
                    0.1, 0.8, 0.3, 0.1,
                    help="Nếu độ liên quan thấp hơn ngưỡng này, hệ thống sẽ sử dụng AI"
                )
                max_images = st.slider("Số ảnh tối đa", 1, 8, 5)
                show_sources = st.checkbox("Hiển thị tài liệu nguồn", value=True)
                intelligent_embedding = st.checkbox("Chèn ảnh thông minh bằng AI", value=True, 
                                                  help="AI sẽ quyết định vị trí chèn ảnh phù hợp")
                
                st.markdown("### 🎨 Tính năng AI")
                st.write("- 🧠 AI phân tích nội dung để chèn ảnh")
                st.write("- 📍 Xác định vị trí tối ưu cho ảnh")
                st.write("- 🔄 Fallback thông minh khi cần")
                st.write("- 📱 Responsive image layout")
            else:
                st.warning("⚠️ Database trống")
                st.info("Hệ thống sẽ sử dụng AI để trả lời")
                st.markdown("""
                **Để sử dụng chế độ RAG:**
                1. Chạy `collect.py` để thu thập dữ liệu
                2. Đảm bảo thư mục `db` có dữ liệu
                3. Khởi động lại ứng dụng
                """)
                relevance_threshold = 0.3
                max_images = 5
                show_sources = True
                intelligent_embedding = True
        
        # Main interface
        st.markdown("### 💬 Đặt câu hỏi")
        question = st.text_area(
            "Nhập câu hỏi của bạn:",
            placeholder="Ví dụ: Hãy mô tả chi tiết về các sản phẩm trong ảnh và giải thích cách sử dụng chúng..." if rag_system.has_data 
                       else "Ví dụ: Hãy giải thích về trí tuệ nhân tạo và ứng dụng của nó...",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            ask_button = st.button("🚀 Hỏi AI", type="primary", use_container_width=True)
        
        if question and ask_button:
            if rag_system.has_data:
                # Tìm kiếm với fallback thông minh
                search_result, relevant_docs, text_context, relevant_images, similarity_score = enhanced_search_with_fallback(
                    rag_system, question, relevance_threshold
                )
                
                # Hiển thị trạng thái tìm kiếm
                display_search_status(search_result, similarity_score, relevance_threshold)
                
                if search_result == "success":
                    # Sử dụng database với AI thông minh
                    with st.spinner("🤔 AI đang phân tích và tạo câu trả lời với ảnh minh họa..."):
                        if intelligent_embedding:
                            # Sử dụng AI để chèn ảnh thông minh
                            answer, images_used = ask_gemini_with_intelligent_images(
                                question,
                                text_context,
                                relevant_images[:max_images],
                                config["GEMINI_API_KEY"],
                                config["MODEL_NAME"]
                            )
                            
                            st.markdown("## 💡 Trả lời (AI + Database với ảnh thông minh)")
                            smart_display_answer_with_embedded_images(answer, images_used)
                        else:
                            # Sử dụng phương pháp tự động
                            answer, images_used = ask_gemini_with_intelligent_images(
                                question,
                                text_context,
                                relevant_images[:max_images],
                                config["GEMINI_API_KEY"],
                                config["MODEL_NAME"]
                            )
                            
                            st.markdown("## 💡 Trả lời (Database + ảnh tự động)")
                            auto_embed_images_in_answer(answer, images_used)
                        
                        if show_sources:
                            with st.expander("📚 Tài liệu nguồn"):
                                for i, doc in enumerate(relevant_docs, 1):
                                    st.write(f"**{i}. {doc['title']}**")
                                    st.write(f"**URL:** {doc['url']}")
                                    st.write(f"**Mô tả:** {doc['description']}")
                                    st.write(f"**Số ảnh:** {len(doc['images'])}")
                                    st.write("---")
                else:
                    # Fallback to AI
                    with st.spinner("🤔 AI đang suy nghĩ..."):
                        answer = ask_gemini_direct(
                            question, 
                            config["GEMINI_API_KEY"], 
                            config["MODEL_NAME"]
                        )
                        st.markdown("## 💡 Trả lời (từ AI)")
                        st.markdown(answer)
                        
                        if search_result == "no_results":
                            st.info("💡 Câu trả lời này được tạo bởi AI vì không tìm thấy thông tin liên quan trong database.")
                        elif search_result == "low_relevance":
                            st.info("💡 Câu trả lời này được tạo bởi AI vì thông tin trong database không đủ liên quan.")
                        elif search_result == "error":
                            st.info("💡 Câu trả lời này được tạo bởi AI do lỗi khi truy cập database.")
            else:
                # Không có database - sử dụng AI
                st.info("📝 Sử dụng AI để trả lời (không có database)")
                
                with st.spinner("🤔 AI đang suy nghĩ..."):
                    answer = ask_gemini_direct(
                        question, 
                        config["GEMINI_API_KEY"], 
                        config["MODEL_NAME"]
                    )
                    st.markdown("## 💡 Trả lời (từ AI)")
                    st.markdown(answer)
                    st.info("💡 Để có câu trả lời dựa trên dữ liệu cụ thể với ảnh minh họa, hãy thu thập dữ liệu vào database.")
    
    except Exception as e:
        st.error(f"Lỗi khởi tạo ứng dụng: {e}")
        st.code(traceback.format_exc())
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 Hệ thống RAG Đa phương tiện với AI - Chèn ảnh thông minh
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
