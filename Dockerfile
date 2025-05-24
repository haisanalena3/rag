# Sử dụng Python 3.11 làm base image
FROM python:3.11-slim

# Thiết lập metadata
LABEL maintainer="Your Name" \
      version="1.0.0" \
      description="RAG Multimodal App with Streamlit"

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các dependencies hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Tạo thư mục db để lưu dữ liệu
RUN mkdir -p db

# Tạo user không phải root để bảo mật
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup appuser && \
    chown -R appuser:appgroup /app

# Expose port cho Streamlit
EXPOSE 8501

# Chuyển sang user không phải root
USER appuser

# Thiết lập environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Chạy ứng dụng Streamlit với các tham số chính xác
ENTRYPOINT ["streamlit", "run", "src/app_local.py", "--server.port=8501", "--server.address=0.0.0.0"]
