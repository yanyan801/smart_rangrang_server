"""
Phase 4 - Memory Manager: SQLite 对话存储 + TF-IDF 语义检索 + 用户画像
双层记忆 —— 短期（LLM上下文）+ 长期（SQLite持久化 + 语义索引）
"""

import json
import logging
import os
import pickle
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# === TF-IDF 分词配置 ===
# char_wb + ngram_range(2,4) 适合中文: 同时捕捉双字词、三字词、四字词
TFIDF_KWARGS = {
    "analyzer": "char_wb",
    "ngram_range": (2, 4),
    "max_features": 2000,
}


class ChatStore:
    """SQLite 对话记录持久化"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chat_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    interrupted INTEGER DEFAULT 0,
                    turn_index INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    topics TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_session_id
                    ON chat_records(session_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON chat_records(timestamp);
            """)

    def save_turn(self, session_id: str, role: str, content: str,
                  turn_index: int, interrupted: int = 0):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_records (session_id, role, content, timestamp, interrupted, turn_index) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, time.time(), interrupted, turn_index),
            )

    def load_recent_turns(self, limit: int = 20) -> list[dict]:
        """加载最近 N 轮对话（跨 session）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, content, session_id FROM chat_records "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        # 反转回时间顺序
        return [dict(r) for r in reversed(rows)]

    def load_session_turns(self, session_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, content FROM chat_records WHERE session_id=? "
                "ORDER BY turn_index",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_turn_count(self, session_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(turn_index) FROM chat_records WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return (row[0] or 0) + 1

    def save_summary(self, session_id: str, summary: str, topics: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO session_summaries (session_id, summary, topics, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (session_id, summary, topics, time.time()),
            )

    def get_all_summaries(self) -> list[dict]:
        """获取所有 session 摘要，用于重建语义索引"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT session_id, summary, topics FROM session_summaries "
                "ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]


class MemoryIndex:
    """
    TF-IDF + 余弦相似度语义检索。
    替代 ChromaDB，零额外依赖，适合中文会话摘要检索。
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
        self._texts: list[str] = []
        self._session_ids: list[str] = []
        self._matrix = None
        self._dirty = False

    def add(self, text: str, session_id: str):
        self._texts.append(text)
        self._session_ids.append(session_id)
        self._dirty = True

    def _rebuild(self):
        if not self._dirty or not self._texts:
            return
        self._matrix = self.vectorizer.fit_transform(self._texts)
        self._dirty = False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义检索 top-k 相关摘要"""
        if not self._texts:
            return []

        self._rebuild()

        if not query.strip():
            # 空查询 → 返回最近摘要
            results = []
            for i in range(min(top_k, len(self._texts))):
                idx = len(self._texts) - 1 - i
                results.append({
                    "summary": self._texts[idx],
                    "session_id": self._session_ids[idx],
                    "score": 1.0,
                })
            return results

        query_vec = self.vectorizer.transform([query])
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.05:  # 最低相似度阈值
                results.append({
                    "summary": self._texts[idx],
                    "session_id": self._session_ids[idx],
                    "score": float(scores[idx]),
                })
        return results

    def rebuild_from_db(self, summaries: list[dict]):
        """从数据库重建索引"""
        self._texts = []
        self._session_ids = []
        for s in summaries:
            self._texts.append(s["summary"])
            self._session_ids.append(s["session_id"])
        self._dirty = True
        if self._texts:
            self._rebuild()
            logger.info(f"MemoryIndex rebuilt: {len(self._texts)} summaries")

    def save(self, path: str):
        """持久化索引到磁盘"""
        self._rebuild()
        data = {
            "texts": self._texts,
            "session_ids": self._session_ids,
            "vocabulary": self.vectorizer.vocabulary_,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """从磁盘加载索引"""
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._texts = data["texts"]
        self._session_ids = data["session_ids"]
        self.vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
        self.vectorizer.vocabulary_ = data.get("vocabulary", {})
        self._dirty = True
        self._rebuild()
        logger.info(f"MemoryIndex loaded: {len(self._texts)} summaries")


class UserProfile:
    """JSON 用户画像"""

    def __init__(self, profile_path: str):
        self.path = profile_path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "name": "",
            "preferred_name": "",
            "interests": [],
            "common_topics": [],
            "response_style": "friendly",  # friendly / concise / detailed
            "total_sessions": 0,
            "total_turns": 0,
            "first_seen": "",
            "last_seen": "",
        }

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def record_session(self):
        self._data["total_sessions"] += 1
        self._data["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self._data["first_seen"]:
            self._data["first_seen"] = self._data["last_seen"]
        self.save()

    def record_turn(self):
        self._data["total_turns"] += 1
        self.save()

    def update_name(self, name: str):
        self._data["preferred_name"] = name
        self.save()

    def add_topic(self, topic: str):
        if topic not in self._data["common_topics"]:
            self._data["common_topics"].append(topic)
            if len(self._data["common_topics"]) > 20:
                self._data["common_topics"] = self._data["common_topics"][-20:]
            self.save()

    def get_context_text(self) -> str:
        """生成注入 LLM 的用户画像文本"""
        d = self._data
        parts = []
        if d["preferred_name"]:
            parts.append(f"用户称呼: {d['preferred_name']}")
        if d["interests"]:
            parts.append(f"用户兴趣: {', '.join(d['interests'][-5:])}")
        if d["common_topics"]:
            parts.append(f"常聊话题: {', '.join(d['common_topics'][-5:])}")
        if d["response_style"]:
            parts.append(f"回复偏好: {d['response_style']}")

        if parts:
            return "## 用户画像\n" + "\n".join(parts) + "\n"
        return ""


class MemoryManager:
    """
    记忆管理器：协调 ChatStore + MemoryIndex + UserProfile

    用法:
        mgr = MemoryManager("E:/.../data")
        mgr.start_session()

        # 每轮对话
        mgr.save_turn("user", user_text)
        mgr.save_turn("assistant", llm_response)

        # 获取记忆上下文（注入 LLM system prompt）
        context = mgr.get_memory_context(user_query)

        # 会话结束
        await mgr.end_session(llm_engine)
    """

    def __init__(self, data_dir: str = ""):
        if not data_dir:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        db_path = os.path.join(data_dir, "chat_history.db")
        index_path = os.path.join(data_dir, "memory_index.pkl")
        profile_path = os.path.join(data_dir, "user_profile.json")

        self.chat = ChatStore(db_path)
        self.index = MemoryIndex()
        self.profile = UserProfile(profile_path)

        # 从 DB 重建语义索引
        summaries = self.chat.get_all_summaries()
        if summaries:
            self.index.rebuild_from_db(summaries)

        # 尝试加载持久化索引
        self.index.load(index_path)

        self._session_id = ""
        self._turn_index = 0

    def start_session(self) -> str:
        """开始新会话，返回 session_id"""
        self._session_id = uuid.uuid4().hex[:12]
        self._turn_index = 0
        self.profile.record_session()
        logger.info(f"Memory: new session {self._session_id}")
        return self._session_id

    def get_history_for_llm(self) -> list[dict]:
        """
        获取应注入 LLM 的历史对话。
        加载最近 20 轮跨 session 对话。
        """
        recent = self.chat.load_recent_turns(20)
        history = []
        for r in recent:
            history.append({"role": r["role"], "content": r["content"]})
        return history

    def get_memory_context(self, query: str = "") -> str:
        """
        构建记忆上下文文本，供注入 LLM system prompt。
        包含: 用户画像 + 相关历史摘要。
        """
        parts = []

        # 用户画像
        profile_text = self.profile.get_context_text()
        if profile_text:
            parts.append(profile_text)

        # 语义检索相关历史摘要
        if query:
            memories = self.index.search(query, top_k=3)
            if memories:
                memory_lines = ["## 相关历史记忆"]
                for i, m in enumerate(memories):
                    memory_lines.append(f"{i+1}. {m['summary']}")
                parts.append("\n".join(memory_lines))

        return "\n".join(parts) if parts else ""

    def save_turn(self, role: str, content: str, interrupted: int = 0):
        """保存一轮对话到 SQLite"""
        if not self._session_id:
            return
        if not content.strip():
            return

        self.chat.save_turn(
            self._session_id, role, content, self._turn_index, interrupted
        )

        if role == "assistant":
            self._turn_index += 1
            self.profile.record_turn()

    async def end_session(self, llm_engine=None) -> str:
        """
        结束会话，生成摘要并存入语义索引。
        返回生成的摘要文本。
        """
        if not self._session_id:
            return ""

        turns = self.chat.load_session_turns(self._session_id)
        if len(turns) < 2:
            self._session_id = ""
            return ""

        # 合成对话文本用于摘要
        conversation_text = ""
        for t in turns[-10:]:  # 最近 10 条
            role_label = "用户" if t["role"] == "user" else "助手"
            conversation_text += f"{role_label}: {t['content']}\n"

        summary = ""
        if llm_engine:
            try:
                summary = await self._generate_summary(llm_engine, conversation_text)
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")

        if not summary:
            summary = self._simple_summary(turns)

        if summary:
            self.chat.save_summary(self._session_id, summary)
            self.index.add(summary, self._session_id)
            # 持久化索引
            index_path = os.path.join(os.path.dirname(self.chat.db_path), "memory_index.pkl")
            self.index.save(index_path)
            logger.info(f"Memory: summary saved for {self._session_id}")

        self._session_id = ""
        self._turn_index = 0
        return summary

    async def _generate_summary(self, llm_engine, conversation_text: str) -> str:
        """使用 LLM 生成对话摘要"""
        prompt = (
            "请用1-2句简短中文总结以下对话的核心内容，提取关键话题和用户偏好：\n\n"
            f"{conversation_text}\n\n"
            "简短摘要："
        )
        summary = ""
        async for token in llm_engine.chat_stream(prompt):
            summary += token
            if len(summary) > 200:
                break
        return summary.strip()

    def _simple_summary(self, turns: list[dict]) -> str:
        """无 LLM 时的简单摘要：取用户首句"""
        for t in turns:
            if t["role"] == "user" and len(t["content"]) > 3:
                return f"用户讨论了: {t['content'][:80]}"
        return ""

    def reset(self):
        """重置当前会话（不结束，不生成摘要）"""
        self._session_id = ""
        self._turn_index = 0
