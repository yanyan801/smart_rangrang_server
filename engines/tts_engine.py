"""
Phase 2 - TTS 引擎: Edge-TTS 流式合成
"""

import asyncio
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class TTSEngine:
    """Edge-TTS 流式语音合成引擎"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural",
                 rate: str = "+15%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self._current_task: asyncio.Task | None = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        流式合成文本，逐个 yield PCM 音频字节。
        每块约 4KB，16kHz/16bit/mono PCM。
        """
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        self._current_task = asyncio.current_task()

        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    # 词边界事件，可用于唇形同步（暂不使用）
                    pass
        except asyncio.CancelledError:
            logger.info("TTS synthesis cancelled")
            raise

    async def flush(self):
        """取消当前合成"""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            self._current_task = None
