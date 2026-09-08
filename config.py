"""从 config.json 读取统一配置，供各阶段脚本复用。

用法：
    import config
    config.get("chunking.chunk_size")
    config.path("paths.data_dir")   # 相对路径自动基于项目根目录解析
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

_CONFIG: dict = {}
if CONFIG_PATH.exists():
    _CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def get(key: str, default=None):
    """按点号路径取配置，如 get("chunking.chunk_size")；缺失时返回 default。"""
    node = _CONFIG
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def path(key: str, default=None):
    """取路径配置并解析为 Path；相对路径基于项目根目录。"""
    v = get(key, default)
    if v is None:
        return None
    p = Path(v)
    return p if p.is_absolute() else BASE_DIR / p


def headers():
    """切分标题配置：[[mark, key], ...] -> [(mark, key), ...]。"""
    return [tuple(h) for h in get("chunking.headers_to_split_on", [])]
