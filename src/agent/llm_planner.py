"""
Temporal Query Decomposer & LLM Planner
Phan ra truy van co yeu to thoi gian tuan tu thanh cac sub-queries doc lap.
Primary: Groq API (llama-3.1-8b-instant). Fallback: Ollama local (qwen2.5:7b).
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional

import yaml
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TemporalPlanner")

TEMPORAL_SYSTEM_PROMPT = """You are a query analyzer for a video retrieval system.
Your ONLY task: determine if the user's query describes a TEMPORAL SEQUENCE (events in chronological order) and decompose it into ordered sub-queries.

Temporal markers include (Vietnamese & English): "trước khi", "sau khi", "rồi", "sau đó", "tiếp theo", "trước", "sau", "before", "after", "then", "followed by", "next", "first...then", "đầu tiên...sau đó".

RULES:
1. If the query contains a temporal sequence, set is_temporal=true and split into ordered sub-queries.
2. If NOT temporal, set is_temporal=false and return the original query as a single sub-query.
3. Each sub-query must be a self-contained visual description (remove temporal connectors).
4. Return ONLY valid JSON. No explanation, no markdown fences.

OUTPUT FORMAT:
{"is_temporal": boolean, "sub_queries": [{"step": 1, "query": "..."}, {"step": 2, "query": "..."}], "original_query": "..."}

EXAMPLES:
Input: "người đàn ông uống nước trước khi ăn cơm"
Output: {"is_temporal": true, "sub_queries": [{"step": 1, "query": "người đàn ông uống nước"}, {"step": 2, "query": "người đàn ông ăn cơm"}], "original_query": "người đàn ông uống nước trước khi ăn cơm"}

Input: "xe ô tô màu đỏ trên đường phố"
Output: {"is_temporal": false, "sub_queries": [{"step": 1, "query": "xe ô tô màu đỏ trên đường phố"}], "original_query": "xe ô tô màu đỏ trên đường phố"}

Input: "a person opens a door then walks into a room"
Output: {"is_temporal": true, "sub_queries": [{"step": 1, "query": "a person opens a door"}, {"step": 2, "query": "a person walks into a room"}], "original_query": "a person opens a door then walks into a room"}"""


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _parse_json_response(raw_text: str, original_query: str) -> Dict[str, Any]:
    """Parse LLM output to structured dict. Fallback: regex extraction."""
    fallback = {
        "is_temporal": False,
        "sub_queries": [{"step": 1, "query": original_query}],
        "original_query": original_query,
    }

    cleaned = raw_text.strip()
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Attempt 1: Direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "sub_queries" in parsed:
            parsed.setdefault("original_query", original_query)
            parsed.setdefault("is_temporal", False)
            if isinstance(parsed["sub_queries"], list) and len(parsed["sub_queries"]) > 0:
                return parsed
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract first JSON object via brace matching
    brace_depth = 0
    start_idx = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                try:
                    parsed = json.loads(cleaned[start_idx : i + 1])
                    if isinstance(parsed, dict) and "sub_queries" in parsed:
                        parsed.setdefault("original_query", original_query)
                        parsed.setdefault("is_temporal", False)
                        return parsed
                except json.JSONDecodeError:
                    pass
                break

    # Attempt 3: Regex extraction of sub_queries array
    sq_match = re.search(
        r'"sub_queries"\s*:\s*(\[.*?\])', cleaned, re.DOTALL
    )
    is_temp_match = re.search(r'"is_temporal"\s*:\s*(true|false)', cleaned, re.IGNORECASE)

    if sq_match:
        try:
            sub_queries = json.loads(sq_match.group(1))
            is_temporal = is_temp_match.group(1).lower() == "true" if is_temp_match else False
            return {
                "is_temporal": is_temporal,
                "sub_queries": sub_queries,
                "original_query": original_query,
            }
        except json.JSONDecodeError:
            pass

    logger.warning("JSON parsing failed for LLM response. Returning non-temporal fallback.")
    return fallback


class TemporalQueryDecomposer:
    """Phan ra truy van thoi gian tuan tu qua LLM (Groq primary, Ollama fallback)."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        llm_cfg = self.config.get("models", {}).get("llm", {})

        groq_key_raw = os.environ.get("GROQ_API_KEY", "") or llm_cfg.get("groq_api_key", "")
        self.groq_api_keys: List[str] = [
            k.strip() for k in groq_key_raw.split(",") if k.strip() and k.strip() != "YOUR_GROQ_API_KEY_HERE"
        ]
        self._groq_key_idx = 0
        self.groq_model = llm_cfg.get("groq_model", "llama-3.1-8b-instant")

        self.ollama_url = llm_cfg.get("ollama_url", "http://localhost:11434")
        self.ollama_model = llm_cfg.get("model_name", "qwen2.5:7b-instruct-q4_K_M")

    def _next_groq_key(self) -> Optional[str]:
        if not self.groq_api_keys:
            return None
        key = self.groq_api_keys[self._groq_key_idx % len(self.groq_api_keys)]
        self._groq_key_idx += 1
        return key

    def _call_groq(self, user_query: str) -> Optional[str]:
        api_key = self._next_groq_key()
        if not api_key:
            return None

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": TEMPORAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            timeout=8,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, user_query: str) -> Optional[str]:
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": TEMPORAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def decompose(self, user_query: str) -> Dict[str, Any]:
        """Phan ra truy van thanh cac sub-queries co thu tu thoi gian."""
        raw_text = None

        # Primary: Groq API
        try:
            raw_text = self._call_groq(user_query)
            if raw_text:
                logger.info("Temporal decomposition via Groq API succeeded.")
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}. Falling back to Ollama.")

        # Fallback: Ollama local
        if not raw_text:
            try:
                raw_text = self._call_ollama(user_query)
                if raw_text:
                    logger.info("Temporal decomposition via Ollama fallback succeeded.")
            except Exception as e:
                logger.warning(f"Ollama fallback also failed: {e}. Returning non-temporal default.")

        if not raw_text:
            return {
                "is_temporal": False,
                "sub_queries": [{"step": 1, "query": user_query}],
                "original_query": user_query,
            }

        return _parse_json_response(raw_text, user_query)


class LLMPlanner:
    """Wrapper: dieu phoi TemporalQueryDecomposer va tra ve search strategy."""

    def __init__(self, config_path: str = "config.yaml"):
        self.decomposer = TemporalQueryDecomposer(config_path=config_path)

    def plan_search_strategy(self, query: str) -> Dict[str, Any]:
        """Phan tich truy van va tra ve ke hoach tim kiem co/khong co yeu to thoi gian."""
        temporal_plan = self.decomposer.decompose(query)

        strategy = {
            "original_query": query,
            "is_temporal": temporal_plan["is_temporal"],
            "sub_queries": temporal_plan["sub_queries"],
        }

        if temporal_plan["is_temporal"]:
            logger.info(
                f"Temporal query detected: {len(temporal_plan['sub_queries'])} steps identified."
            )
        else:
            logger.info("Non-temporal query. Single-pass retrieval strategy.")

        return strategy


if __name__ == "__main__":
    planner = LLMPlanner()
    test_queries = [
        "người đàn ông uống nước trước khi ăn cơm",
        "xe ô tô màu đỏ trên đường phố",
        "a child runs across the field then falls down",
    ]
    for q in test_queries:
        result = planner.plan_search_strategy(q)
        print(json.dumps(result, ensure_ascii=False, indent=2))
