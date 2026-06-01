"""
Phase 2 - ASR 引擎: SenseVoice-small 封装
支持 VAD 分句后的离线识别 + 简单流式(按句输出)
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ASREngine:
    """SenseVoice-small ASR 引擎"""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model: Optional[object] = None
        self._load_time = 0.0

    def load(self):
        """加载模型"""
        from funasr import AutoModel

        logger.info(f"Loading SenseVoice-small (device={self.device})...")
        t0 = time.time()
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            device=self.device,
            disable_update=True,
        )
        self._load_time = time.time() - t0
        logger.info(f"ASR model loaded in {self._load_time:.1f}s")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def transcribe(self, audio: "numpy.ndarray", sample_rate: int = 16000) -> str:
        """
        识别一段完整语音，返回文本。
        audio: float32 numpy array, shape (n_samples,)
        """
        import numpy as np

        if not self.is_loaded:
            raise RuntimeError("ASR model not loaded")

        # 确保是 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 归一化
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        result = self.model.generate(
            input=audio,
            language="zh",
            use_itn=True,
            batch_size_s=60,
        )
        if result and len(result) > 0:
            text = result[0].get("text", "")
            return _clean_tags(text)
        return ""

    def transcribe_wav(self, wav_path: str) -> str:
        """识别 WAV 文件"""
        if not self.is_loaded:
            raise RuntimeError("ASR model not loaded")

        result = self.model.generate(
            input=wav_path,
            language="zh",
            use_itn=True,
        )
        if result and len(result) > 0:
            text = result[0].get("text", "")
            return _clean_tags(text)
        return ""


def _clean_tags(text: str) -> str:
    """清理 SenseVoice 特殊标签 <|...|>"""
    import re
    return re.sub(r'<\|[^|]*\|>', '', text).strip()
