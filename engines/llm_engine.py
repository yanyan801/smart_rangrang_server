"""
Phase 4 - LLM 引擎: Ollama 流式对话 + 记忆上下文注入
"""

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个友好的语音助手，名叫小智。
你的回答要求：
1. 简洁，适合语音播报，每次回复不超过100字
2. 自然口语化，像朋友聊天
3. 如果用户问题模糊，可以简短追问"""


class LLMEngine:
    """Ollama 流式对话引擎（支持记忆上下文）"""

    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 model: str = "qwen2.5:14b"):
        self.base_url = base_url
        self.model = model
        self._history: list[dict] = []
        self._memory_context: str = ""

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/tags", timeout=5)
                return resp.status_code == 200
        except Exception:
            return False

    # === 记忆接口 ===

    def set_memory_context(self, context: str):
        """设置记忆上下文（用户画像 + 相关历史摘要），注入 system prompt"""
        self._memory_context = context

    def load_history_from_memory(self, history: list[dict]):
        """从 MemoryManager 预加载历史对话"""
        self._history = history.copy()
        logger.info(f"LLM: loaded {len(history)} history turns from memory")

    def add_to_history(self, role: str, content: str):
        self._history.append({"role": role, "content": content})
        if len(self._history) > 30:
            self._history = self._history[-30:]

    def reset_history(self):
        self._history = []

    # === 流式对话 ===

    async def chat_stream(self, user_text: str) -> AsyncIterator[str]:
        """
        流式对话: 逐个 yield token 文本。
        memory_context 自动注入 system prompt。
        """
        system_content = SYSTEM_PROMPT
        if self._memory_context:
            system_content = (
                f"{SYSTEM_PROMPT}\n\n"
                f"## 上下文记忆\n{self._memory_context}"
            )

        messages = [
            {"role": "system", "content": system_content},
            *self._history,
            {"role": "user", "content": user_text},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
            },
        }

        full_response = ""

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "message" in chunk and chunk["message"].get("content"):
                        token = chunk["message"]["content"]
                        full_response += token
                        yield token

                    if chunk.get("done"):
                        break

        # 对话完成后加入历史
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": full_response})
        if len(self._history) > 30:
            self._history = self._history[-30:]

    def cancel(self):
        if self._history and self._history[-1]["role"] == "user":
            pass  # 保留 user 消息，assistant 回复未完成不入 history
