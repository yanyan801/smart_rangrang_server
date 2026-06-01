"""
Phase 5 - Watchdog: 超时守卫 + 会话看门狗 + 健康监控
防卡死三层体制 —— 组件超时 / 全局看门狗 / 异常自愈
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# === 超时配置 ===
ASR_TIMEOUT_MS = 10_000        # ASR 单次识别超时
LLM_FIRST_TOKEN_MS = 15_000    # LLM 首 token 超时
LLM_TOTAL_MS = 60_000          # LLM 总生成超时
TTS_CHUNK_MS = 5_000           # TTS 单句合成超时
SESSION_IDLE_MS = 30_000       # 会话空闲超时 → 回 IDLE
HEARTBEAT_INTERVAL_MS = 3_000  # 心跳间隔
HEARTBEAT_TIMEOUT_MS = 8_000   # 心跳超时 → 断连
MAX_CONSECUTIVE_ERRORS = 3     # 连续错误上限


class TimeoutGuard:
    """异步超时守卫 —— 在超时后执行回调"""

    def __init__(self, timeout_ms: float, name: str = ""):
        self.timeout_ms = timeout_ms
        self.name = name
        self._task: asyncio.Task | None = None
        self._triggered = False

    async def __aenter__(self):
        self._triggered = False
        self._task = asyncio.create_task(self._sleep())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return False  # 不抑制异常

    async def _sleep(self):
        await asyncio.sleep(self.timeout_ms / 1000)
        self._triggered = True

    @property
    def triggered(self) -> bool:
        return self._triggered


async def with_timeout(coro, timeout_ms: float, name: str = ""):
    """在超时时间内执行协程，超时返回 None"""
    try:
        async with TimeoutGuard(timeout_ms, name) as guard:
            result = await coro
        if guard.triggered:
            logger.warning(f"Timeout [{name}]: {timeout_ms}ms exceeded")
            return None
        return result
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Timeout guard error [{name}]: {e}")
        return None


@dataclass
class HealthTracker:
    """组件健康状态追踪"""

    name: str
    ok_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_ok: float = 0.0
    last_error: float = 0.0
    last_error_msg: str = ""
    total_timeouts: int = 0

    def report_ok(self):
        self.ok_count += 1
        self.consecutive_errors = 0
        self.last_ok = time.time()

    def report_error(self, msg: str = ""):
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_error = time.time()
        self.last_error_msg = msg

    def report_timeout(self):
        self.total_timeouts += 1
        self.consecutive_errors += 1
        self.last_error = time.time()

    @property
    def is_healthy(self) -> bool:
        return self.consecutive_errors < MAX_CONSECUTIVE_ERRORS

    @property
    def is_degraded(self) -> bool:
        """连续错误 >= 上限 → 服务降级"""
        return self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS


class SessionWatchdog:
    """
    会话级看门狗 —— 空闲超时 + 心跳 + 错误计数

    用法:
        wd = SessionWatchdog(on_idle_timeout=go_idle, on_strike_out=shutdown)
        wd.reset()           # 每次有活动时重置
        wd.check_heartbeat() # 收到心跳时调用
        wd.report_error("asr_timeout")
    """

    def __init__(self,
                 idle_timeout_ms: float = SESSION_IDLE_MS,
                 heartbeat_timeout_ms: float = HEARTBEAT_TIMEOUT_MS,
                 max_strikes: int = MAX_CONSECUTIVE_ERRORS,
                 on_idle_timeout=None,
                 on_heartbeat_lost=None,
                 on_strike_out=None):
        self.idle_timeout_ms = idle_timeout_ms
        self.heartbeat_timeout_ms = heartbeat_timeout_ms
        self.max_strikes = max_strikes
        self._on_idle_timeout = on_idle_timeout
        self._on_heartbeat_lost = on_heartbeat_lost
        self._on_strike_out = on_strike_out

        self._last_activity = time.time()
        self._last_heartbeat = time.time()
        self._strike_count = 0
        self._heartbeat_lost = False

    def reset(self):
        """有活动 → 重置空闲计时器和心跳计时器"""
        self._last_activity = time.time()
        self._last_heartbeat = time.time()
        self._heartbeat_lost = False

    def heartbeat(self):
        """收到心跳"""
        self._last_heartbeat = time.time()
        self._heartbeat_lost = False

    def report_error(self, component: str = ""):
        """报告一次错误"""
        self._strike_count += 1
        logger.warning(
            f"Watchdog: strike {self._strike_count}/{self.max_strikes} "
            f"({component})"
        )

    def report_ok(self):
        """报告一次正常 → 重置错误计数"""
        self._strike_count = 0

    @property
    def is_idle_timeout(self) -> bool:
        return self._idle_ms > self.idle_timeout_ms

    @property
    def is_heartbeat_lost(self) -> bool:
        return self._heartbeat_ms > self.heartbeat_timeout_ms

    @property
    def is_strike_out(self) -> bool:
        return self._strike_count >= self.max_strikes

    @property
    def _idle_ms(self) -> float:
        return (time.time() - self._last_activity) * 1000

    @property
    def _heartbeat_ms(self) -> float:
        return (time.time() - self._last_heartbeat) * 1000

    def check(self) -> str:
        """
        检查看门狗状态，返回触发的事件名或空字符串。

        返回: "idle" | "heartbeat_lost" | "strike_out" | ""
        """
        if self.is_strike_out:
            if self._on_strike_out:
                self._on_strike_out()
            return "strike_out"

        if self.is_heartbeat_lost and not self._heartbeat_lost:
            self._heartbeat_lost = True
            if self._on_heartbeat_lost:
                self._on_heartbeat_lost()
            return "heartbeat_lost"

        if self.is_idle_timeout:
            if self._on_idle_timeout:
                self._on_idle_timeout()
            return "idle"

        return ""


class ComponentHealth:
    """全局组件健康管理"""

    def __init__(self):
        self.components: dict[str, HealthTracker] = {}

    def get(self, name: str) -> HealthTracker:
        if name not in self.components:
            self.components[name] = HealthTracker(name=name)
        return self.components[name]

    def report_ok(self, name: str):
        self.get(name).report_ok()

    def report_error(self, name: str, msg: str = ""):
        self.get(name).report_error(msg)

    def report_timeout(self, name: str):
        self.get(name).report_timeout()

    @property
    def all_healthy(self) -> bool:
        return all(c.is_healthy for c in self.components.values())

    def get_degraded(self) -> list[str]:
        return [c.name for c in self.components.values() if c.is_degraded]

    def status_text(self) -> str:
        lines = []
        for name, c in self.components.items():
            status = "OK" if c.is_healthy else "DEGRADED"
            lines.append(
                f"  {name}: {status} "
                f"(ok={c.ok_count}, err={c.error_count}, "
                f"consec={c.consecutive_errors}, to={c.total_timeouts})"
            )
        return "\n".join(lines) if lines else "  (no components)"
