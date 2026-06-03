"""
Phase 2 - VAD 引擎: 能量阈值 VAD（简化版，Phase 2 验证用）
生产环境替换为 Silero VAD (ONNX on Pi)
"""

import time
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class VADEngine:
    """能量阈值 VAD + 句子边界检测"""

    def __init__(self, sample_rate: int = 16000,
                 energy_threshold: float = 0.02,
                 min_speech_frames: int = 3,
                 min_silence_frames: int = 8,
                 max_speech_s: float = 8.0):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.max_speech_s = max_speech_s
        self.state = _VADState()

    def load(self):
        """能量 VAD 无需加载模型"""
        logger.info("Energy-based VAD ready (no model needed)")

    @property
    def is_loaded(self) -> bool:
        return True

    def _compute_energy(self, audio: np.ndarray) -> float:
        """计算归一化 RMS 能量"""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return np.sqrt(np.mean(audio ** 2))

    def process_frame(self, audio_chunk: np.ndarray) -> float:
        """
        返回归一化能量值 (0-1 近似)。
        audio_chunk: int16 numpy array
        """
        return self._compute_energy(audio_chunk)

    def update(self, energy: float, audio_chunk: np.ndarray):
        """
        更新 VAD 状态，返回 (event, audio) 元组：
          - ("speech_start", None)
          - ("speech_end", np.ndarray of int16)
          - (None, None)
        """
        state = self.state

        if energy >= self.energy_threshold:
            # 语音帧
            state.accumulated_audio.append(audio_chunk)
            if not state.is_speaking:
                state.speech_frames += 1
                if state.speech_frames >= self.min_speech_frames:
                    state.is_speaking = True
                    state.speech_start_time = time.time()
                    state.silence_frames = 0
                    logger.debug("VAD: speech_start")
                    return "speech_start", None
            else:
                state.silence_frames = 0
                # 最长说话时长保护：超过 max_speech_s 强制断句
                if self.max_speech_s > 0:
                    dur = time.time() - state.speech_start_time
                    if dur >= self.max_speech_s:
                        return self._end_speech()
        else:
            # 静音帧
            if state.is_speaking:
                state.silence_frames += 1
                state.accumulated_audio.append(audio_chunk)
                if state.silence_frames >= self.min_silence_frames:
                    return self._end_speech()
            else:
                # 持续静音: 保留缓冲（捕捉句首）
                state.accumulated_audio.append(audio_chunk)
                max_buf = int(self.sample_rate * 1.5 / len(audio_chunk))
                if len(state.accumulated_audio) > max_buf:
                    state.accumulated_audio = state.accumulated_audio[-max_buf:]

        return None, None

    @property
    def last_silence_ms(self) -> float:
        """最后一次 speech_end 的静音时长（ms）"""
        return self.state.last_silence_ms

    def reset(self):
        self.state = _VADState()

    def _end_speech(self):
        """结束当前语音段，返回累积音频"""
        state = self.state
        state.is_speaking = False
        state.speech_end_time = time.time()
        state.last_silence_ms = state.silence_frames * 40  # 每帧 40ms
        combined = np.concatenate(state.accumulated_audio)
        duration = len(combined) / self.sample_rate
        state.accumulated_audio = []
        state.speech_frames = 0
        reason = "silence" if state.silence_frames >= self.min_silence_frames else "max_duration"
        logger.info(f"VAD: speech_end ({duration:.1f}s, reason={reason})")
        return "speech_end", combined


class _VADState:
    """VAD 内部状态"""
    def __init__(self):
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.speech_end_time = 0.0
        self.speech_frames = 0
        self.silence_frames = 0
        self.accumulated_audio: list = []
        self.last_silence_ms = 0.0
