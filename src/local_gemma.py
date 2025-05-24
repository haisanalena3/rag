import requests
import json
import time
import logging

logger = logging.getLogger(__name__)

class LocalGemmaClient:
    def __init__(self, config):
        # Đảm bảo base_url đúng format
        self.base_url = str(config["base_url"]).rstrip('/')
        self.model = str(config["model"])
        self.max_tokens = config.get("max_tokens", 2000)
        self.temperature = config.get("temperature", 0.4)
        self.top_p = config.get("top_p", 0.4)
        self.timeout = config.get("timeout", 220)
        
        # Validate URL format
        if not self.base_url.startswith(('http://', 'https://')):
            self.base_url = f"https://{self.base_url}"
        
        logger.info(f"Khởi tạo LocalGemmaClient với URL: {self.base_url}")
        logger.info(f"Model: {self.model}")
        
    def check_connection(self):
        """Kiểm tra kết nối tới Ollama server"""
        try:
            # Thử endpoint health check
            health_url = f"{self.base_url}"
            logger.info(f"Kiểm tra kết nối tới: {health_url}")
            
            response = requests.get(health_url, timeout=5)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response content: {response.text[:200]}")
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Lỗi kiểm tra kết nối: {e}")
            return False
    
    def get_model_info(self):
        """Lấy thông tin model hiện tại"""
        try:
            url = f"{self.base_url}/api/show"
            payload = {"name": self.model}
            
            logger.info(f"Gọi API show model: {url}")
            logger.info(f"Payload: {payload}")
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API show model trả về status: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin model: {e}")
            return None
        
    def generate_response(self, prompt, context="", max_tokens=None):
        """Tạo response từ Gemma model"""
        try:
            # Tạo full prompt với context
            if context:
                full_prompt = f"{context}\n\nCâu hỏi: {prompt}\n\nTrả lời:"
            else:
                full_prompt = prompt
            
            # Tạo URL đúng cách
            api_url = f"{self.base_url}/api/generate"
            logger.info(f"Gọi API generate: {api_url}")
            
            # Chuẩn bị payload cho API
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": max_tokens or self.max_tokens
                }
            }
            
            logger.info(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            # Gọi API với headers đúng
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Response JSON: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            return result.get("response", "Không thể tạo phản hồi")
            
        except requests.exceptions.Timeout:
            error_msg = "⏰ Yêu cầu đã hết thời gian chờ. Vui lòng thử lại với câu hỏi ngắn gọn hơn."
            logger.error(error_msg)
            return error_msg
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Lỗi kết nối đến AI model: {str(e)}"
            logger.error(f"RequestException: {e}")
            return error_msg
            
        except Exception as e:
            error_msg = f"❌ Đã xảy ra lỗi: {str(e)}"
            logger.error(f"Lỗi không xác định: {e}")
            return error_msg
    
    def generate_chat_response(self, messages, max_tokens=None):
        """Tạo response cho chat format (OpenAI compatible)"""
        try:
            api_url = f"{self.base_url}/v1/chat/completions"
            logger.info(f"Gọi API chat: {api_url}")
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Lỗi chat API: {e}")
            # Fallback về generate API
            prompt = messages[-1]["content"] if messages else ""
            context = "\n".join([msg["content"] for msg in messages[:-1]]) if len(messages) > 1 else ""
            return self.generate_response(prompt, context, max_tokens)
    
    def test_model(self):
        """Test model với câu hỏi đơn giản"""
        try:
            test_response = self.generate_response("Xin chào! Bạn có thể giới thiệu về bản thân không?")
            return True, test_response
        except Exception as e:
            return False, str(e)
