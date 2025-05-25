import requests
import json
from typing import Optional, Dict, Any, List
import time
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalGemmaClient:
    def __init__(self, base_url="http://localhost:11434", model="gemma3:4b"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 2000, 
                         temperature: float = 0.7, top_p: float = 0.9, timeout: int = 120) -> str:
        """Tạo response từ Gemma 3n local"""
        try:
            # Kết hợp context và prompt
            if context.strip():
                full_prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi một cách chi tiết và chính xác:

Thông tin tham khảo:
{context}

Câu hỏi: {prompt}

Hướng dẫn trả lời:
1. Hãy trả lời bằng tiếng Việt
2. Dựa trên thông tin đã cung cấp
3. Nếu có đề cập đến hình ảnh, sử dụng cú pháp [IMAGE_X] để chỉ định vị trí ảnh
4. Trả lời một cách có cấu trúc và dễ hiểu
5. Nếu thông tin không đủ, hãy nói rõ điều đó

Trả lời:"""
            else:
                full_prompt = f"""Câu hỏi: {prompt}

Hãy trả lời câu hỏi này bằng tiếng Việt một cách chi tiết và chính xác."""
            
            logger.info(f"Gửi request đến Ollama với model: {self.model}")
            
            # Gọi Ollama API
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                        "stop": ["Human:", "Assistant:", "User:", "Question:", "Câu hỏi:"]
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "Không thể tạo câu trả lời.")
                logger.info("Nhận được response thành công từ Ollama")
                return generated_text.strip()
            else:
                error_msg = f"Lỗi API: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Lỗi: Timeout khi gọi model local. Vui lòng thử lại hoặc giảm độ dài câu hỏi."
            logger.error(error_msg)
            return error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "Lỗi: Không thể kết nối với Ollama server. Vui lòng kiểm tra xem Ollama có đang chạy không."
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Lỗi kết nối với Gemma local: {e}"
            logger.error(error_msg)
            return error_msg
    
    def check_connection(self) -> bool:
        """Kiểm tra kết nối với Ollama"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Lấy danh sách models có sẵn"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []
    
    def check_model_exists(self, model_name: str) -> bool:
        """Kiểm tra model có tồn tại không"""
        available_models = self.get_available_models()
        return any(model_name in model for model in available_models)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Lấy thông tin chi tiết về model"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def pull_model(self, model_name: str) -> bool:
        """Tải model về local"""
        try:
            logger.info(f"Đang tải model {model_name}...")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 phút timeout cho việc tải model
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Lỗi khi tải model {model_name}: {e}")
            return False
