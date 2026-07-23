"""
Query Expansion Agent (Qwen2.5-7B qua Ollama API)
Mở rộng truy vấn tiếng Việt, sinh từ đồng nghĩa và từ khóa tìm kiếm.
"""

from typing import List
import requests


class QueryExpansionAgent:
    """
    Gọi LLM Qwen2.5-7B (GGUF INT4 qua Ollama) để mở rộng từ khóa tìm kiếm.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "qwen2.5:7b-instruct-q4_K_M"):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def expand_query(self, user_query: str) -> List[str]:
        """
        Mở rộng user_query thành danh sách câu/từ khóa đồng nghĩa.
        """
        prompt = (
            f"Bạn là hệ thống mở rộng truy vấn cho AI Challenge retrieval.\n"
            f"Hãy phân tích truy vấn người dùng: '{user_query}'\n"
            f"Hãy trả về 3-5 câu hoặc từ khóa tương đương bằng tiếng Việt."
        )
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=5
            )
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                expansions = [line.strip() for line in result_text.split("\n") if line.strip()]
                return expansions if expansions else [user_query]
        except Exception as e:
            print(f"[QueryExpansion] LLM Call failed: {e}")
        
        return [user_query]
