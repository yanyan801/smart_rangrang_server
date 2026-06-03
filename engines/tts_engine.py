"""
TTS 引擎: Edge-TTS 流式合成 + MP3→PCM 解码
输出 16kHz / 16bit / mono PCM，每帧 640 samples (40ms / 1280 bytes)
"""

import asyncio
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 640 * 2   # 40ms @ 16kHz/16bit = 1280 bytes


class TTSEngine:
    """Edge-TTS 流式语音合成引擎（输出 PCM）"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural",
                 rate: str = "+15%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self._current_task: asyncio.Task | None = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """流式合成 + MP3解码，逐个 yield 1280-byte PCM chunks。"""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        self._current_task = asyncio.current_task()
        mp3_buf = bytearray()
        pcm_buf = bytearray()
        loop = asyncio.get_event_loop()

        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buf.extend(chunk["data"])

                    try:
                        pcm = await loop.run_in_executor(
                            None, _decode_mp3, bytes(mp3_buf)
                        )
                    except Exception:
                        continue  # 解码失败，继续积累 MP3

                    mp3_buf.clear()
                    pcm_buf.extend(pcm)

                    # 输出完整的 PCM chunks
                    while len(pcm_buf) >= _CHUNK_BYTES:
                        chunk_out = bytes(pcm_buf[:_CHUNK_BYTES])
                        del pcm_buf[:_CHUNK_BYTES]
                        yield chunk_out

            # 流结束：flush 残留 MP3
            if mp3_buf:
                try:
                    pcm = await loop.run_in_executor(
                        None, _decode_mp3, bytes(mp3_buf)
                    )
                    pcm_buf.extend(pcm)
                except Exception:
                    pass

            # flush 残留 PCM
            while len(pcm_buf) >= _CHUNK_BYTES:
                chunk_out = bytes(pcm_buf[:_CHUNK_BYTES])
                del pcm_buf[:_CHUNK_BYTES]
                yield chunk_out

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


def _decode_mp3(data: bytes) -> bytes:
    """MP3 → PCM bytes (16kHz / 16bit / mono)"""
    import miniaudio

    decoded = miniaudio.decode(
        data,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=16000,
    )
    return bytes(decoded.samples)
