"""
Smart RangRang - 启动入口
用法:
    python start.py                    # 加载 config.yaml
    python start.py --config prod.yaml # 指定配置文件
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import uvicorn

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import smart_rangrang_server.server as svr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("startup")


def main():
    parser = argparse.ArgumentParser(description="Smart RangRang Server")
    parser.add_argument(
        "--config", "-c",
        default=str(Path(__file__).parent / "config.yaml"),
        help="配置文件路径（默认 smart_rangrang_server/config.yaml）",
    )
    parser.add_argument(
        "--host",
        help="绑定地址（覆盖配置文件）",
    )
    parser.add_argument(
        "--port", type=int,
        help="端口（覆盖配置文件）",
    )
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    svr.set_config(config)

    # 初始化引擎
    logger.info("Loading models...")
    svr.load_engines(config)
    svr.asr.load()
    svr.vad.load()
    logger.info("All models loaded.")

    # 启动服务
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    logger.info(f"Starting server on http://{host}:{port}")
    uvicorn.run(svr.app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
