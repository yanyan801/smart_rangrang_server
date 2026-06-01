"""
Phase 3 - Echo Judge: 回声判决器
双模判决 —— 互相关 + 能量包络，区分真实人声与扬声器回声

原理:
  扬声器播放 TTS → 麦克风回采 → 如果 AEC 未完全消除 → VAD 可能误判为"有人说话"
  Echo Judge 对比麦克风信号与扬声器参考信号:
    - 互相关峰值高 → 回声
    - 能量包络相似 → 回声
    - 都不满足 → 真人说话

用法:
    judge = EchoJudge()
    is_echo = judge.judge(mic_chunk, speaker_ref_chunk)
"""

import logging
from collections import deque

import numpy as np
from scipy.signal import correlate

logger = logging.getLogger(__name__)


class EchoJudge:
    """回声判决器"""

    def __init__(self,
                 sample_rate: int = 16000,
                 corr_threshold: float = 0.5,         # 强相关阈值
                 corr_weak_threshold: float = 0.3,     # 弱相关阈值（配合能量）
                 env_threshold: float = 0.6,           # 能量包络相似度阈值
                 ref_buffer_seconds: float = 2.0,      # 参考信号缓冲时长
                 chunk_ms: int = 40):                  # 处理帧长
        self.sample_rate = sample_rate
        self.corr_threshold = corr_threshold
        self.corr_weak_threshold = corr_weak_threshold
        self.env_threshold = env_threshold
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

        # 扬声器参考信号环形缓冲（最近 N 秒的播放音频）
        max_len = int(sample_rate * ref_buffer_seconds)
        self._ref_buffer = deque(maxlen=max_len)

    def feed_reference(self, audio_chunk: np.ndarray):
        """
        喂入播放中的 TTS 音频作为参考信号。
        audio_chunk: int16 numpy array, 任意长度
        """
        self._ref_buffer.extend(audio_chunk.tolist())

    def judge(self, mic_chunk: np.ndarray) -> tuple[bool, float, float]:
        """
        判决麦克风信号是否为回声。

        mic_chunk: int16 numpy array, 一帧音频 (如 640 samples / 40ms)

        返回: (is_echo, cross_corr_peak, energy_ratio)
          - is_echo=True  → 是回声，不触发打断
          - is_echo=False → 不是回声，可能是真人说话
        """
        if len(self._ref_buffer) < len(mic_chunk):
            # 参考信号不足（刚开始播放），保守处理：不判为回声
            return False, 0.0, 0.0

        # 取最近一段参考信号（同样长度）
        ref = np.array(
            [self._ref_buffer[i] for i in range(
                len(self._ref_buffer) - len(mic_chunk),
                len(self._ref_buffer)
            )],
            dtype=np.float32
        )
        mic = mic_chunk.astype(np.float32)

        # 1. 归一化互相关峰值
        mic_norm = np.linalg.norm(mic)
        ref_norm = np.linalg.norm(ref)
        if mic_norm < 1e-6 or ref_norm < 1e-6:
            return False, 0.0, 0.0

        corr = correlate(mic, ref, mode='full')
        peak = float(np.max(np.abs(corr))) / (mic_norm * ref_norm)

        # 2. 能量包络对比
        mic_env = float(np.mean(np.abs(mic)))
        ref_env = float(np.mean(np.abs(ref)))
        energy_ratio = min(mic_env, ref_env) / max(mic_env, ref_env + 1e-10)

        # 3. 综合判决
        is_echo = False
        if peak > self.corr_threshold:
            # 强相关 → 回声
            is_echo = True
        elif peak > self.corr_weak_threshold and energy_ratio > self.env_threshold:
            # 中等相关 + 能量接近 → 回声
            is_echo = True

        if is_echo:
            logger.debug(
                f"EchoJudge: ECHO (peak={peak:.3f}, env_ratio={energy_ratio:.3f})"
            )

        return is_echo, peak, energy_ratio

    def flush(self):
        """清空参考缓冲（打断后调用）"""
        self._ref_buffer.clear()


class InterruptDetector:
    """
    打断检测器：VAD + Echo Judge 双模 AND 逻辑

    打断 = VAD 检测到连续语音 AND Echo Judge 判断不是回声
    """

    def __init__(self,
                 echo_judge: EchoJudge,
                 vad_consecutive_frames: int = 10,   # 连续语音帧确认
                 vad_energy_threshold: float = 0.04):  # 语音能量阈值
        self.echo_judge = echo_judge
        self.vad_consecutive_frames = vad_consecutive_frames
        self.vad_energy_threshold = vad_energy_threshold
        self._speech_counter = 0
        self._echo_counter = 0

    def process(self, mic_chunk: np.ndarray, energy: float) -> bool:
        """
        处理一帧麦克风音频，返回是否触发打断。

        返回: True = 真人打断，应触发 interrupt
        """
        if energy < self.vad_energy_threshold:
            # 能量太低 → 静音 → 重置计数器
            self._speech_counter = 0
            self._echo_counter = 0
            return False

        is_echo, peak, _ = self.echo_judge.judge(mic_chunk)

        if is_echo:
            self._speech_counter = 0
            self._echo_counter += 1
        else:
            self._speech_counter += 1
            self._echo_counter = 0

        if self._speech_counter >= self.vad_consecutive_frames:
            # 连续 N 帧检测到非回声语音 → 触发打断
            logger.info(
                f"InterruptDetector: TRIGGER (speech_frames={self._speech_counter})"
            )
            self._speech_counter = 0
            return True

        return False

    def reset(self):
        """重置计数器"""
        self._speech_counter = 0
        self._echo_counter = 0
