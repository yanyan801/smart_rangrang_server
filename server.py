"""
Smart RangRang - WebSocket Server
全功能集成: VAD + ASR + LLM + TTS + 打断检测 + 智能断句 + 聊天记忆 + 稳定性看门狗

用法:
    python start.py                    # 加载 config.yaml 启动
    python start.py --config prod.yaml # 指定配置文件
"""

import asyncio
import json
import logging
import os
import time
import traceback

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .engines.stt_engine import ASREngine
from .engines.llm_engine import LLMEngine
from .engines.tts_engine import TTSEngine
from .engines.vad_engine import VADEngine
from .core.echo_judge import EchoJudge, InterruptDetector
from .core.sentence_judge import SentenceJudge
from .core.memory_manager import MemoryManager
from .core.watchdog import SessionWatchdog, ComponentHealth

logger = logging.getLogger("server")

# === 延迟初始化的全局引擎（由 load_engines() 填充） ===
asr: ASREngine = None
llm: LLMEngine = None
tts: TTSEngine = None
vad: VADEngine = None
echo_judge: EchoJudge = None
interrupt_detector: InterruptDetector = None
sentence_judge: SentenceJudge = None
memory: MemoryManager = None
health: ComponentHealth = None

# 运行时参数（从 config 加载）
SAMPLE_RATE = 16000
CHUNK_MS = 40
CHUNK_SAMPLES = 640

IDLE = "IDLE"
LISTEN = "LISTEN"
THINK = "THINK"
SPEAK = "SPEAK"

app = FastAPI()


def load_engines(config: dict):
    """根据配置初始化所有引擎（在 start.py 中调用）"""
    global asr, llm, tts, vad, echo_judge, interrupt_detector
    global sentence_judge, memory, health
    global SAMPLE_RATE, CHUNK_MS, CHUNK_SAMPLES

    cfg = config
    SAMPLE_RATE = cfg["server"]["sample_rate"]
    CHUNK_MS = cfg["server"]["chunk_ms"]
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

    asr = ASREngine(device=cfg["asr"]["device"])
    llm = LLMEngine(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
    )
    tts = TTSEngine(
        voice=cfg["tts"]["voice"],
        rate=cfg["tts"]["rate"],
        pitch=cfg["tts"]["pitch"],
    )
    vad = VADEngine(
        sample_rate=SAMPLE_RATE,
        energy_threshold=cfg["vad"]["energy_threshold"],
        min_speech_frames=cfg["vad"]["min_speech_frames"],
        min_silence_frames=cfg["vad"]["min_silence_frames"],
    )

    echo_judge = EchoJudge(
        sample_rate=SAMPLE_RATE,
        corr_threshold=cfg["echo_judge"]["corr_threshold"],
        corr_weak_threshold=cfg["echo_judge"]["corr_weak_threshold"],
        env_threshold=cfg["echo_judge"]["env_threshold"],
        ref_buffer_seconds=cfg["echo_judge"]["ref_buffer_seconds"],
        chunk_ms=CHUNK_MS,
    )

    interrupt_detector = InterruptDetector(
        echo_judge=echo_judge,
        vad_consecutive_frames=cfg["interrupt"]["vad_consecutive_frames"],
        vad_energy_threshold=cfg["interrupt"]["vad_energy_threshold"],
    )

    sentence_judge = SentenceJudge(
        punct_silence_ms=cfg["sentence"]["punct_silence_ms"],
        force_silence_ms=cfg["sentence"]["force_silence_ms"],
        long_text_chars=cfg["sentence"]["long_text_chars"],
        long_text_silence_ms=cfg["sentence"]["long_text_silence_ms"],
        incomplete_extend_ms=cfg["sentence"]["incomplete_extend_ms"],
        pending_timeout_ms=cfg["sentence"]["pending_timeout_ms"],
    )

    data_dir = os.path.join(os.path.dirname(__file__), cfg["data"]["dir"])
    memory = MemoryManager(data_dir)

    health = ComponentHealth()
    for name in ["asr", "llm", "tts", "interrupt_detector", "audio_processing", "server"]:
        health.get(name)

    logger.info("=== Smart RangRang Server ===")
    logger.info(
        f"Config: ASR to={cfg['asr']['timeout_ms']}ms, "
        f"LLM first={cfg['llm']['first_token_timeout_ms']}ms "
        f"total={cfg['llm']['total_timeout_ms']}ms, "
        f"TTS={cfg['tts']['chunk_timeout_ms']}ms, "
        f"idle={cfg['watchdog']['session_idle_ms']}ms, "
        f"heartbeat={cfg['watchdog']['heartbeat_interval_ms']}ms"
    )


# === HTTP 端点 ===
@app.get("/")
async def test_page():
    return HTMLResponse(_TEST_PAGE)


@app.get("/health")
async def health_check():
    if health is None:
        return {"status": "initializing"}
    return {
        "status": "ok" if health.all_healthy else "degraded",
        "components": {
            name: {
                "healthy": c.is_healthy,
                "ok_count": c.ok_count,
                "error_count": c.error_count,
                "consecutive_errors": c.consecutive_errors,
            }
            for name, c in health.components.items()
        },
    }


# === WebSocket ===
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    wd_cfg = _config("watchdog")
    asr_cfg = _config("asr")
    llm_cfg = _config("llm")
    tts_cfg = _config("tts")

    # 会话初始化
    session_id = memory.start_session()
    llm.reset_history()
    history = memory.get_history_for_llm()
    if history:
        llm.load_history_from_memory(history)
    memory_ctx = memory.get_memory_context()
    if memory_ctx:
        llm.set_memory_context(memory_ctx)

    session = {"state": IDLE}
    tts_task: list[asyncio.Task] = []
    llm_cancel: list[asyncio.Event] = []

    # 看门狗
    wd = SessionWatchdog(
        idle_timeout_ms=wd_cfg["session_idle_ms"],
        heartbeat_timeout_ms=wd_cfg["heartbeat_timeout_ms"],
        max_strikes=wd_cfg["max_consecutive_errors"],
    )
    heartbeat_task = asyncio.create_task(
        _send_heartbeat(ws, wd, wd_cfg["heartbeat_interval_ms"])
    )

    try:
        while True:
            # 看门狗检查
            wd_event = wd.check()
            if wd_event == "idle" and session["state"] != IDLE:
                logger.info("Session idle → IDLE")
                await _cleanup(session, tts_task, llm_cancel)
                session["state"] = IDLE
            elif wd_event == "strike_out":
                await ws.send_json({
                    "type": "state", "state": "degraded",
                    "message": "服务繁忙，稍等哦",
                })

            # 接收消息（带断连保护）
            try:
                data = await ws.receive()
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.info(f"Client disconnected: {e}")
                break

            if "bytes" in data:
                raw = data["bytes"]
                if len(raw) != CHUNK_SAMPLES * 2:
                    continue

                audio_chunk = np.frombuffer(raw, dtype=np.int16).copy()
                s = session["state"]
                wd.reset()

                if s == SPEAK:
                    try:
                        energy = vad.process_frame(audio_chunk)
                        if interrupt_detector.process(audio_chunk, energy):
                            logger.info("INTERRUPT")
                            await _interrupt(ws, session, tts_task, llm_cancel)
                            continue
                    except Exception as e:
                        logger.error(f"Interrupt error: {e}")
                        health.report_error("interrupt_detector", str(e))

                if s in (LISTEN, IDLE):
                    try:
                        await _process_audio(
                            ws, audio_chunk, session, tts_task, llm_cancel,
                            asr_cfg, llm_cfg, tts_cfg,
                        )
                    except Exception as e:
                        logger.error(f"Audio error: {e}")
                        health.report_error("audio_processing", str(e))
                        wd.report_error("audio_processing")
                        session["state"] = LISTEN

                # 断句器超时
                if session["state"] == LISTEN and sentence_judge.has_pending:
                    pending = sentence_judge.check_timeout()
                    if pending:
                        logger.info(f"SentenceJudge timeout → '{pending}'")
                        await ws.send_json({
                            "type": "asr_result", "text": pending, "time_s": 0,
                        })
                        memory.save_turn("user", pending)
                        await _llm_tts(
                            ws, pending, llm_cancel, tts_task, session,
                            llm_cfg, tts_cfg,
                        )

            elif "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    wd.heartbeat()
                    await ws.send_json({"type": "pong"})

                elif msg_type == "reset":
                    logger.info("Client reset")
                    await _cleanup(session, tts_task, llm_cancel)
                    await memory.end_session(llm)
                    session_id = memory.start_session()
                    llm.reset_history()
                    history = memory.get_history_for_llm()
                    if history:
                        llm.load_history_from_memory(history)
                    memory_ctx = memory.get_memory_context()
                    if memory_ctx:
                        llm.set_memory_context(memory_ctx)
                    session["state"] = IDLE
                    await ws.send_json({"type": "state", "state": IDLE})

    except (WebSocketDisconnect, RuntimeError):
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected: {e}\n{traceback.format_exc()}")
        health.report_error("server", str(e))
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        await _cleanup(session, tts_task, llm_cancel)
        await memory.end_session(llm)
        logger.info(f"Session ended, final state={session['state']}")


# === 核心处理函数 ===

async def _process_audio(ws, audio_chunk, session, tts_task, llm_cancel,
                         asr_cfg, llm_cfg, tts_cfg):
    energy = vad.process_frame(audio_chunk)
    event, speech_audio = vad.update(energy, audio_chunk)

    if event == "speech_start" and session["state"] == IDLE:
        session["state"] = LISTEN
        logger.info("State: IDLE → LISTEN")

    if event == "speech_end" and session["state"] == LISTEN:
        if speech_audio is None or len(speech_audio) < SAMPLE_RATE * 0.3:
            return

        silence_ms = vad.last_silence_ms
        vad.reset()
        session["state"] = THINK
        logger.info(
            f"State: LISTEN → THINK "
            f"(audio={len(speech_audio)/SAMPLE_RATE:.1f}s, silence={silence_ms:.0f}ms)"
        )

        # ASR（带超时）
        t_asr = time.time()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(asr.transcribe, speech_audio.astype(np.float32)),
                timeout=asr_cfg["timeout_ms"] / 1000,
            )
            health.report_ok("asr")
        except asyncio.TimeoutError:
            logger.warning(f"ASR timeout")
            health.report_timeout("asr")
            session["state"] = LISTEN
            return
        except Exception as e:
            logger.error(f"ASR error: {e}")
            health.report_error("asr", str(e))
            session["state"] = LISTEN
            return

        t_asr = time.time() - t_asr
        logger.info(f"ASR [{t_asr:.2f}s]: {text}")

        if not text.strip():
            session["state"] = LISTEN
            return

        await ws.send_json({
            "type": "asr_result", "text": text, "time_s": round(t_asr, 2),
        })

        is_complete, judged_text = sentence_judge.judge(text, silence_ms)
        if not is_complete:
            logger.info(f"SentenceJudge: buffering '{judged_text[-20:]}'")
            session["state"] = LISTEN
            return

        memory.save_turn("user", judged_text)
        memory_ctx = memory.get_memory_context(judged_text)
        if memory_ctx:
            llm.set_memory_context(memory_ctx)

        await _llm_tts(ws, judged_text, llm_cancel, tts_task, session,
                       llm_cfg, tts_cfg)


async def _llm_tts(ws, text, llm_cancel, tts_task, session, llm_cfg, tts_cfg):
    t_llm_start = time.time()
    first_token = True
    llm_text = ""
    t_first_token = 0.0
    llm_timeout = False

    cancel_evt = asyncio.Event()
    llm_cancel.clear()
    llm_cancel.append(cancel_evt)

    async def stream():
        nonlocal first_token, llm_text, t_first_token, llm_timeout
        try:
            async for token in llm.chat_stream(text):
                if cancel_evt.is_set():
                    break

                if first_token:
                    t_first_token = time.time() - t_llm_start
                    ttft_ms = t_first_token * 1000
                    logger.info(f"LLM first token: {ttft_ms:.0f}ms")
                    if ttft_ms > llm_cfg["first_token_timeout_ms"]:
                        logger.warning(f"LLM first token slow ({ttft_ms:.0f}ms)")
                        health.report_timeout("llm")
                    first_token = False

                llm_text += token

                elapsed = (time.time() - t_llm_start) * 1000
                if elapsed > llm_cfg["total_timeout_ms"]:
                    logger.warning(f"LLM total timeout")
                    llm_timeout = True
                    health.report_timeout("llm")
                    break

            if llm_timeout and not llm_text.strip():
                llm_text = "我正在想，请稍等..."

            if llm_text.strip() and not cancel_evt.is_set():
                health.report_ok("llm")
                session["state"] = SPEAK
                logger.info(f"State: THINK → SPEAK | LLM: {llm_text[:50]}...")
                await ws.send_json({
                    "type": "llm_result", "text": llm_text,
                    "ttft_ms": round(t_first_token * 1000),
                })

                chunk_count = 0
                t_tts_start = time.time()
                try:
                    async for audio_bytes in tts.synthesize_stream(llm_text):
                        if cancel_evt.is_set():
                            break
                        chunk_elapsed = (time.time() - t_tts_start) * 1000
                        if chunk_elapsed > tts_cfg["chunk_timeout_ms"] and chunk_count == 0:
                            logger.warning("TTS first chunk timeout")
                            health.report_timeout("tts")
                            break

                        ref_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                        echo_judge.feed_reference(ref_chunk)
                        await ws.send_bytes(audio_bytes)
                        chunk_count += 1
                        await asyncio.sleep(0)

                    health.report_ok("tts")
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    health.report_error("tts", str(e))

                await ws.send_json({"type": "tts_end"})
                logger.info(f"TTS done: {chunk_count} chunks")
                memory.save_turn("assistant", llm_text)
                session["state"] = LISTEN
                logger.info("State: SPEAK → LISTEN")
        except asyncio.CancelledError:
            if llm_text.strip():
                memory.save_turn("assistant", llm_text, interrupted=1)
            logger.info("LLM/TTS cancelled")

    task = asyncio.create_task(stream())
    tts_task.clear()
    tts_task.append(task)


async def _interrupt(ws, session, tts_task, llm_cancel):
    session["state"] = IDLE
    if llm_cancel:
        llm_cancel[0].set()
    if tts_task and not tts_task[0].done():
        tts_task[0].cancel()
    await ws.send_json({"type": "stop_playback"})
    echo_judge.flush()
    vad.reset()
    interrupt_detector.reset()
    sentence_judge.reset()
    session["state"] = LISTEN
    logger.info("State: SPEAK → LISTEN (interrupted)")


async def _cleanup(session, tts_task, llm_cancel):
    if llm_cancel:
        llm_cancel[0].set()
    if tts_task and not tts_task[0].done():
        tts_task[0].cancel()
    echo_judge.flush()
    vad.reset()
    interrupt_detector.reset()
    sentence_judge.reset()


async def _send_heartbeat(ws, wd, interval_ms):
    try:
        while True:
            await asyncio.sleep(interval_ms / 1000)
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break
    except asyncio.CancelledError:
        pass


def _config(section: str) -> dict:
    """从加载的配置中获取子配置（需要在 load_engines 后调用）"""
    # 运行时从全局变量推导，或从 _loaded_config 获取
    return _loaded_config.get(section, {})


# === 配置引用（load_engines 后设置） ===
_loaded_config: dict = {}


def set_config(config: dict):
    global _loaded_config
    _loaded_config = config


# === 测试页面 ===
_TEST_PAGE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Smart RangRang</title></head>
<body>
<h2>Smart RangRang 语音助手</h2>
<button id="start">开始录音</button>
<button id="stop" disabled>停止</button>
<span id="status">就绪</span>
<span id="heartbeat" style="color:green">●</span>
<hr>
<div><b>ASR:</b> <span id="asr"></span></div>
<div><b>LLM:</b> <span id="llm"></span></div>
<div><b>状态:</b> <span id="state"></span></div>
<div><b>心跳:</b> <span id="hb"></span></div>
<hr>
<div id="log"></div>

<script>
let ws, mediaStream, audioCtx, processor, source;
let isRecording = false, lastHb = Date.now();

document.getElementById('start').onclick = async () => {
    ws = new WebSocket('ws://' + location.host + '/ws');
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) return;
        const msg = JSON.parse(e.data);
        if (msg.type === 'asr_result') {
            document.getElementById('asr').textContent = msg.text;
        } else if (msg.type === 'llm_result') {
            document.getElementById('llm').textContent = msg.text;
            document.getElementById('state').textContent = 'SPEAK';
        } else if (msg.type === 'tts_end') {
            document.getElementById('state').textContent = 'LISTEN';
        } else if (msg.type === 'stop_playback') {
            document.getElementById('state').textContent = 'LISTEN (打断)';
        } else if (msg.type === 'ping') {
            lastHb = Date.now(); ws.send(JSON.stringify({type: 'ping'}));
        }
        if (msg.type !== 'ping' && msg.type !== 'pong')
            document.getElementById('log').innerHTML =
                '<div>[' + new Date().toLocaleTimeString() + '] ' +
                msg.type + ': ' + (msg.text || '') + '</div>' +
                document.getElementById('log').innerHTML.slice(0, 5000);
    };

    mediaStream = await navigator.mediaDevices.getUserMedia({audio: {sampleRate: 16000, channelCount: 1}});
    audioCtx = new AudioContext({sampleRate: 16000});
    source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(640, 1, 1);
    processor.onaudioprocess = (e) => {
        if (!isRecording || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++)
            int16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
        ws.send(int16.buffer);
    };
    source.connect(processor); processor.connect(audioCtx.destination);
    isRecording = true;
    document.getElementById('start').disabled = true;
    document.getElementById('stop').disabled = false;
    document.getElementById('status').textContent = '录音中...';
    setInterval(() => {
        let e = (Date.now() - lastHb) / 1000;
        document.getElementById('hb').textContent = e.toFixed(1) + 's';
        document.getElementById('heartbeat').style.color = e > 8 ? 'red' : 'green';
    }, 1000);
};

document.getElementById('stop').onclick = () => {
    isRecording = false;
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    ws.send(JSON.stringify({type: 'reset'}));
    document.getElementById('start').disabled = false;
    document.getElementById('stop').disabled = true;
    document.getElementById('status').textContent = '已停止';
};
</script>
</body>
</html>"""
