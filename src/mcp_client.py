import asyncio
import json
import logging
from typing import Tuple, Dict, Any, Optional
import uuid
from mcp import ClientSession
from mcp.client.sse import sse_client

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPClient:
    """Client để giao tiếp với MCP server qua SSE transport"""
    
    def __init__(self, sse_url: str, timeout: int = 30):
        """
        Khởi tạo MCP client với SSE transport.
        
        Args:
            sse_url (str): URL cho SSE endpoint (e.g., http://localhost:8081/sse)
            timeout (int): Timeout cho các request (seconds)
        """
        self.sse_url = sse_url.rstrip('/')
        self.timeout = float(timeout)
        self.session = None
        self._streams_context = None
        self._session_context = None
        self.client_id = str(uuid.uuid4())
        self.tools = []
        self.is_connected = False

    async def connect(self, retries: int = 3) -> bool:
        """Kết nối đến MCP server qua SSE với retries"""
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Đang kết nối đến MCP Server tại {self.sse_url}, thử lần {attempt}...")
                self._streams_context = sse_client(url=self.sse_url, timeout=self.timeout)
                streams = await self._streams_context.__aenter__()
                self._session_context = ClientSession(*streams)
                self.session = await self._session_context.__aenter__()

                await self.session.initialize()
                response = await self.session.list_tools()
                self.tools = response.tools if hasattr(response, 'tools') else []
                self.is_connected = True

                logger.info(f"Đã kết nối thành công! Phát hiện {len(self.tools)} tools")
                for tool in self.tools:
                    logger.info(f"  - {tool.name}: {tool.description}")
                return True

            except Exception as e:
                logger.error(f"Lỗi kết nối đến MCP server (lần {attempt}): {str(e)}")
                self.is_connected = False
                if attempt < retries:
                    await asyncio.sleep(1)
                else:
                    await self.disconnect()
                    return False
        return False

    async def disconnect(self) -> None:
        """Đóng kết nối với MCP server"""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
            self.session = None
            self.is_connected = False
            logger.info(f"Đã đóng kết nối với MCP Server cho client: {self.client_id}")
        except Exception as e:
            logger.error(f"Lỗi khi đóng kết nối: {str(e)}")

    async def test_connection(self) -> Tuple[bool, str]:
        """Kiểm tra kết nối với MCP server"""
        try:
            if not self.is_connected:
                if not await self.connect():
                    return False, "Không thể kết nối đến MCP server"

            response = await self.session.list_tools()
            self.tools = response.tools if hasattr(response, 'tools') else []
            tool_names = [tool.name for tool in self.tools]
            logger.info(f"Tools detected: {tool_names}")
            return True, f"Kết nối thành công với MCP server, phát hiện {len(self.tools)} tools: {', '.join(tool_names)}"

        except Exception as e:
            logger.error(f"Lỗi kiểm tra kết nối: {str(e)}")
            return False, f"Lỗi: {str(e)}"

    async def call_tool(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gọi một tool trên MCP server.
        
        Args:
            method (str): Tên method/tool (e.g., "totalBug")
            params (dict): Tham số cho tool
        
        Returns:
            dict: Kết quả từ tool hoặc None nếu thất bại
        """
        if not self.is_connected:
            if not await self.connect():
                logger.error("Không thể gọi tool: chưa kết nối đến MCP server")
                return None

        try:
            if not any(tool.name == method for tool in self.tools):
                logger.error(f"Tool {method} không tồn tại trong danh sách tools: {[tool.name for tool in self.tools]}")
                return None

            logger.info(f"Gọi tool {method} với tham số: {params}")
            result = await self.session.call_tool(method, params)
            if hasattr(result, 'content'):
                logger.info(f"Kết quả tool {method}: {result.content}")
                return result.content
            logger.error(f"Kết quả tool không hợp lệ: {result}")
            return None

        except Exception as e:
            logger.error(f"Lỗi gọi tool {method}: {str(e)}")
            return None

    async def process_query(self, query: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Xử lý truy vấn bằng cách để AI chọn tool MCP phù hợp.
        
        Args:
            query (str): Câu hỏi từ người dùng
        
        Returns:
            Tuple[Optional[str], Optional[Dict[str, Any]]]: 
                - Prompt gửi đến AI với thông tin tools
                - Kết quả từ tool được chọn (hoặc None nếu không có tool nào được chọn)
        """
        try:
            logger.info(f"Xử lý truy vấn: {query}")

            if not query or not isinstance(query, str):
                logger.error("Truy vấn không hợp lệ: phải là chuỗi không rỗng")
                return "Truy vấn không hợp lệ: phải là chuỗi không rỗng.", None

            if not self.is_connected:
                if not await self.connect():
                    logger.error("Không thể xử lý truy vấn: chưa kết nối đến MCP server")
                    return "Không thể kết nối đến MCP server để tạo prompt.", None

            if not self.tools:
                logger.warning("Không có công cụ nào được phát hiện")
                return "Không có công cụ MCP nào khả dụng để xử lý câu hỏi.", None

            # Tạo prompt với thông tin tools
            tools_info = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.tools
            ])
            prompt = """
            Bạn là một trợ lý thông minh. Dựa trên câu hỏi sau: """ + '"' + query + '"' + """

            Bạn có các công cụ MCP sau để hỗ trợ trả lời:
            """ + tools_info + """

            Hãy chọn công cụ phù hợp nhất để trả lời câu hỏi. Trả về định dạng:

            {
            "tool_name": "totalBug",
            "parameters": {"name": "target_name"}
            }

            Nếu không có công cụ nào phù hợp, hãy trả lời như 1 câu hỏi bình thường.
            Lưu ý: Nếu cần sử dụng công cụ thì chỉ trả về JSON hợp lệ, không thêm bất kỳ văn bản nào khác
            """



            # Trả về prompt để AI xử lý và chọn tool
            logger.info(f"Prompt gửi đến AI: {prompt}")
            return prompt, None

        except Exception as e:
            logger.error(f"Lỗi xử lý truy vấn: {str(e)}")
            return f"Lỗi khi tạo prompt cho MCP tool: {str(e)}", None

    def close(self):
        """Đóng kết nối đồng bộ (cho các ngữ cảnh không async)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.disconnect())
            else:
                loop.run_until_complete(self.disconnect())
        except Exception as e:
            logger.error(f"Lỗi khi đóng kết nối đồng bộ: {str(e)}")