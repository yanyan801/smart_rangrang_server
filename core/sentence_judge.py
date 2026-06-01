"""
Phase 3 - Sentence Judge: 三模式智能断句
结合标点触发、静音时长、语义完整性，判断 ASR 输出是否为完整句子

模式 A - 标点触发: 句末标点（。！？）+ 短静音 → 立即断句
模式 B - 长静音强制: 静音 > 800ms → 强制断句
模式 C - 长文本保护: 文本 > 100 字 + 静音 > 300ms → 强制断句
语义补全: 末尾不完整尾词 → 延长等待至 1500ms
"""

import logging
import time

logger = logging.getLogger(__name__)


class SentenceJudge:
    """三模式智能断句器"""

    # 句末标点（强断句信号）
    SENTENCE_END_PUNCT = set('。！？.!?…~」』）)')

    # 弱断句标点（停顿但不一定断句）
    WEAK_PUNCT = set('，,、：:；;')

    # 不完整尾词：说话者可能还在组织语言，需要延长等待
    INCOMPLETE_TAILS = [
        '那个', '就是', '然后', '还有', '因为', '比如说',
        '嗯', '呃', '这个', '所以', '但是', '不过', '而且',
        '如果', '虽然', '要是', '总之', '反正', '另外',
        '像', '好像', '大概', '可能', '也许', '或者',
        '的话', '的时候', '之后', '以前', '方面',
        '一种', '一个', '一些', '比较', '特别',
    ]

    def __init__(self,
                 punct_silence_ms: float = 200,
                 force_silence_ms: float = 800,
                 long_text_chars: int = 100,
                 long_text_silence_ms: float = 300,
                 incomplete_extend_ms: float = 1500,
                 pending_timeout_ms: float = 5000):
        self.punct_silence_ms = punct_silence_ms
        self.force_silence_ms = force_silence_ms
        self.long_text_chars = long_text_chars
        self.long_text_silence_ms = long_text_silence_ms
        self.incomplete_extend_ms = incomplete_extend_ms
        self.pending_timeout_ms = pending_timeout_ms

        self._pending_text = ""
        self._pending_since = 0.0

    def judge(self, text: str, silence_ms: float) -> tuple[bool, str]:
        """
        判断当前 ASR 文本是否为完整句子。

        Args:
            text: ASR 识别文本
            silence_ms: VAD 检测到的静音时长（ms）

        Returns:
            (is_complete, output_text)
            - is_complete=True: 可以发送给 LLM
            - is_complete=False: 需要等待更多语音（已缓冲）
        """
        text = text.strip()
        if not text:
            return False, ""

        # 合并待处理文本
        if self._pending_text:
            full_text = self._pending_text + text
            logger.debug(f"SentenceJudge: merged pending '{self._pending_text}' + '{text}'")
        else:
            full_text = text

        # === 模式 A: 标点触发 ===
        if self._has_sentence_end(full_text) and silence_ms >= self.punct_silence_ms:
            self._release()
            logger.debug(f"SentenceJudge: Path A (punct, silence={silence_ms:.0f}ms) → '{full_text}'")
            return True, full_text

        # === 模式 C: 长文本保护 ===
        if len(full_text) >= self.long_text_chars and silence_ms >= self.long_text_silence_ms:
            self._release()
            logger.debug(f"SentenceJudge: Path C (long={len(full_text)}chars, silence={silence_ms:.0f}ms) → '{full_text}'")
            return True, full_text

        # === 模式 B: 长静音强制 ===
        if silence_ms >= self.force_silence_ms:
            self._release()
            logger.debug(f"SentenceJudge: Path B (force silence={silence_ms:.0f}ms) → '{full_text}'")
            return True, full_text

        # === 语义补全检查 ===
        if self._has_incomplete_tail(full_text):
            if silence_ms >= self.incomplete_extend_ms:
                self._release()
                logger.debug(f"SentenceJudge: incomplete tail timeout (silence={silence_ms:.0f}ms) → '{full_text}'")
                return True, full_text
            # 缓冲等待更多语音
            self._hold(full_text)
            logger.debug(f"SentenceJudge: incomplete tail '{full_text[-10:]}' → buffering")
            return False, full_text

        # === 默认断句: 中等静音 + 看起来完整 ===
        if silence_ms >= self.punct_silence_ms * 2:
            self._release()
            logger.debug(f"SentenceJudge: default split (silence={silence_ms:.0f}ms) → '{full_text}'")
            return True, full_text

        # 静音太短 → 缓冲
        self._hold(full_text)
        logger.debug(f"SentenceJudge: short silence ({silence_ms:.0f}ms) → buffering")
        return False, full_text

    def check_timeout(self) -> str:
        """
        检查待处理文本是否超时，超时则强制释放。
        应在每次收到音频帧时调用（无论是否有 ASR 结果）。

        Returns:
            超时释放的文本，无超时返回空字符串
        """
        if not self._pending_text:
            return ""
        elapsed = (time.time() - self._pending_since) * 1000
        if elapsed >= self.pending_timeout_ms:
            text = self._pending_text
            self._release()
            logger.debug(f"SentenceJudge: timeout release ({elapsed:.0f}ms) → '{text}'")
            return text
        return ""

    def _has_sentence_end(self, text: str) -> bool:
        """文本是否以句末标点结尾"""
        if not text:
            return False
        return text[-1] in self.SENTENCE_END_PUNCT

    def _has_incomplete_tail(self, text: str) -> bool:
        """文本是否以不完整尾词结尾"""
        for tail in self.INCOMPLETE_TAILS:
            if text.endswith(tail):
                return True
        return False

    def _hold(self, text: str):
        """缓冲待处理文本"""
        self._pending_text = text
        self._pending_since = time.time()

    def _release(self):
        """释放缓冲"""
        self._pending_text = ""
        self._pending_since = 0.0

    @property
    def has_pending(self) -> bool:
        return bool(self._pending_text)

    def flush(self) -> str:
        """强制释放待处理文本（打断/重置时调用）"""
        text = self._pending_text
        self._release()
        return text

    def reset(self):
        self._release()
