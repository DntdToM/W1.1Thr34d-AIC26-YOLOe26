"""
Tree-of-Thoughts (ToT) LLM Planner & Query Expansion Agent
Suy luận nhiều hướng cho câu hỏi phức tạp hoặc truy vấn đa bước (Multi-step / Multi-modal reasoning).
"""

from typing import Dict, Any, List, Optional
import os
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class QueryExpansionAgent:
    """
    QueryExpansionAgent áp dụng kỹ thuật Tree-of-Thoughts (ToT) để phân tách
    truy vấn người dùng thành các sub-queries (Hình ảnh, Âm thanh, Chữ OCR) và gợi ý trọng số.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        llm_cfg = self.config.get("models", {}).get("llm", {})
        self.ollama_url = llm_cfg.get("ollama_url", "http://localhost:11434")
        self.model_name = llm_cfg.get("model_name", "qwen2.5:7b-instruct-q4_K_M")

    def expand_query(self, user_query: str) -> List[str]:
        """
        Khung hàm Mở rộng truy vấn (Query Expansion) bằng Tree-of-Thoughts LLM.

        PROMPT MẪU TREE-OF-THOUGHTS (ToT) PROMPT TEMPLATE HƯỚNG DẪN LLM:
        -----------------------------------------------------------------
        Bạn là một AI Architect chuyên phân tích truy vấn đa phương tiện cho AIC 2026 Retrieval.
        Hãy áp dụng suy luận Tree-of-Thoughts (ToT) qua 3 bước:
        1. Phân tích ngữ cảnh (Thought 1): Xác định các đối tượng thị giác, âm thanh phát ra, 
           và văn bản bảng hiệu xuất hiện trong truy vấn người dùng '{user_query}'.
        2. Tạo các Sub-queries độc lập (Thought 2):
           - visual_subquery: Mô tả hình ảnh (dùng cho SigLIP 2).
           - audio_subquery: Lời nói/giọng nói (dùng cho PhoWhisper).
           - metadata_subquery: Từ khóa OCR/Objects (dùng cho BGE-M3).
        3. Đề xuất Trọng số (Thought 3): Gợi ý bộ trọng số {image: float, audio: float, meta: float} 
           tối ưu cho câu hỏi này.

        Trả về: Danh sách các chuỗi từ khóa mở rộng tương đương.
        """
        pass


class LLMPlanner:
    """
    Tree-of-Thoughts Planner giúp phân tách yêu cầu tìm kiếm phức tạp
    thành các sub-queries (Hình ảnh, Âm thanh, Chữ trong ảnh).
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.agent = QueryExpansionAgent(config_path=config_path)

    def plan_search_strategy(self, query: str) -> Dict[str, Any]:
        """
        Phân tích truy vấn và đưa ra kế hoạch trọng số / truy vấn con.
        """
        pass


if __name__ == "__main__":
    agent = QueryExpansionAgent()
    print("QueryExpansionAgent Skeleton được tạo thành công!")
