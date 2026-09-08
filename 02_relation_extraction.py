import os
import json
import hashlib
from typing import List, Literal, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek

import config

load_dotenv()

## 缺少enntity <----> chunk 的映射

# ========== 1. 图谱 Schema 定义（结构化输出契约） ==========

class Entity(BaseModel):
    """知识图谱中的实体节点"""
    name: str = Field(description="规范化后的实体名称，统一大小写与缩写")
    type: Literal["Method", "Dataset", "Metric", "Task", "ProblemScenario"] = Field(
        description="实体类型"
    )


class Relation(BaseModel):
    """知识图谱中的关系边（head -[relation]-> tail）"""
    head: str = Field(description="头实体名称；若头实体为论文本身，填论文标题")
    relation: Literal[
        "USES", "EVALUATED_ON", "TARGETS",
        "ADDRESSES", "APPLIES_TO", "INVOLVES"
    ] = Field(description="关系类型")
    tail: str = Field(description="尾实体名称")


class Performance(BaseModel):
    """性能结论：方法在某条件/数据集上对某指标的实测数值。"""
    method: str = Field(description="方法名")
    metric: str = Field(description="指标名，如 accuracy、F1")
    value: str = Field(description="具体数值，原样保留，如 90.18、93.8%、12.3 dB")
    condition: str = Field(description="实验条件，如 SNR=10dB、NS=100，无则留空字符串")
    dataset: str = Field(description="数据集名，如 RML2016.10a，无则留空字符串")


class ExtractionResult(BaseModel):
    """单次抽取的结构化结果"""
    entities: List[Entity] = Field(description="抽取出的实体列表")
    relations: List[Relation] = Field(description="抽取出的关系列表")
    performances: List[Performance] = Field(default_factory=list, description="性能结论列表")


# ========== 2. 抽取提示词 ==========

EXTRACTION_PROMPT = """你是论文知识图谱构建助手。请从给定论文文本中抽取实体和关系，构建知识图谱。

【当前论文】{paper_title}

【只抽取以下实体类型】
- Method（方法）：如 BERT、LoRA、对比学习
- Dataset（数据集）：如 ImageNet、SQuAD、RML2016.a
- Metric（指标）：如 F1、BLEU、准确率、参数量
- Task（任务）：如 文本分类、机器翻译、调制识别
- ProblemScenario（问题场景）：如 少样本对抗鲁棒性、低信噪比文本分类
注意：不要抽取 Paper 实体，论文本身已作为已知上下文。

【只抽取以下关系类型】
- USES: (Paper)-[USES]->(Method) 论文用了某方法
- EVALUATED_ON: (Method)-[EVALUATED_ON]->(Dataset) 方法在数据集上评测
- TARGETS: (Paper)-[TARGETS]->(Task) 论文针对的任务
- ADDRESSES: (Paper)-[ADDRESSES]->(ProblemScenario) 论文解决某问题场景
- APPLIES_TO: (Method)-[APPLIES_TO]->(ProblemScenario) 方法适用于某问题场景
- INVOLVES: (ProblemScenario)-[INVOLVES]->(Task) 问题场景涉及某任务

【性能结论（performances）】
- 抽取文本中明确给出的「方法 + 指标 + 具体数值 + 实验条件 + 数据集」的性能结果
- 每条含 method（方法名）、metric（指标名）、value（具体数值）、condition（实验条件）、dataset（数据集）
- value 必须原样保留（如 90.18、93.8%、12.3 dB），不得四舍五入、不得补全、不得臆造
- 同一方法在同一条件/数据集上报告多个指标时，每个指标单独输出一条
- 只抽文本中明确出现的数值，没有数值则 performances 输出空数组

【规范化要求】
- 实体名统一大小写与缩写：如 "BERT-base"、"bert"、"Bidirectional Encoder Representations" 统一规范化为 "BERT"
- 同一实体在全文中只出现一次，不要重复
- 只抽取文本中明确出现的实体，不要臆造或补全
- 关系中的 head 若为论文本身，请填【当前论文】给出的标题，原样填写不要改写

【输出格式】
严格按以下 JSON 结构输出，字段名必须与示例完全一致：
{{
  "entities": [
    {{"name": "LoRA", "type": "Method"}}
  ],
  "relations": [
    {{"head": "论文标题", "relation": "USES", "tail": "LoRA"}}
  ],
  "performances": [
    {{"method": "ResNet", "metric": "accuracy", "value": "90.18", "condition": "SNR=10dB", "dataset": "RML2016.10a"}}
  ]
}}
注意：relations 中表示关系类型的字段名是 "relation"，不要写成 "type"。performances 无内容时输出空数组 []。

【待抽取文本】
{text}
"""


# ========== 3. 路径与模型配置 ==========

DATA_DIR = config.path("paths.data_dir")
CACHE_PATH = config.path("paths.cache_path")

# 缓存 schema 版本：结构变更时 +1，旧条目自动重抽
SCHEMA_VERSION = config.get("extraction.schema_version", 2)

# 抽取任务建议 temperature=0，保证结构化输出稳定
llm = ChatDeepSeek(
    model=config.get("deepseek.model"),
    temperature=config.get("extraction.temperature", 0.0),
    max_tokens=config.get("extraction.max_tokens", 16384),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=config.get("deepseek.base_url"),
)

structured_llm = llm.with_structured_output(ExtractionResult, method="json_mode")

MAX_WORKERS = config.get("extraction.max_workers", 8)  # 并发请求数，触发限流(429)时调低

CHUNK_SIZE = config.get("chunking.chunk_size", 1200)
CHUNK_OVERLAP = config.get("chunking.chunk_overlap", 300)

HEADERS_TO_SPLIT_ON = config.headers()

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=True,  # 标题存入 metadata，正文保持纯净
)

sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def header_prefix(metadata: dict) -> str:
    """从 metadata 重建完整章节标题前缀，保证每个块自带章节上下文。"""
    lines = [
        f"{mark} {metadata[key]}"
        for mark, key in HEADERS_TO_SPLIT_ON
        if metadata.get(key)
    ]
    return "\n".join(lines)


def add_prefix(doc, prefix: str):
    if prefix:
        doc.page_content = f"{prefix}\n{doc.page_content}"
    return doc


def split_markdown(text: str) -> list:
    """按 Markdown 标题切分，长块二次切分，每个块带章节前缀。与 update_index.py 保持一致。"""
    md_docs = markdown_splitter.split_text(text)
    chunks = []
    for doc in md_docs:
        prefix = header_prefix(doc.metadata)
        if len(doc.page_content) <= CHUNK_SIZE:
            chunks.append(add_prefix(doc, prefix))
        else:
            subs = sub_splitter.split_documents([doc])
            for sub in subs:
                chunks.append(add_prefix(sub, prefix))
    return chunks


# ========== 4. 缓存读写 ==========

def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ========== 5. 抽取与去重合并 ===========

def extract_from_text(text: str, paper_title: str) -> ExtractionResult:
    """对单段文本做实体关系抽取。"""
    prompt = EXTRACTION_PROMPT.format(text=text, paper_title=paper_title)
    return structured_llm.invoke(prompt)


def extract_paper(chunks, paper_title: str, max_workers: int = MAX_WORKERS) -> Tuple[Set, Set, list, list]:
    """并发抽取一篇论文的所有 chunk，并按实体 (type, name) 与关系三元组去重。

    同时记录 entity↔chunk 与 performance↔chunk 溯源：
    返回 (entities: {(type, name)}, relations: {(head, relation, tail)},
          mentions: [{name, type, chunk_index}],
          performances: [{method, metric, value, condition, dataset, chunk_index}])
    """
    # 保留原始 chunk 序号（与 03 的 split_markdown 对齐），跳过空白块
    texts = [(i, c.page_content.strip()) for i, c in enumerate(chunks) if c.page_content.strip()]
    entities: Set[Tuple[str, str]] = set()
    relations: Set[Tuple[str, str, str]] = set()
    mentions: list = []
    performances: list = []
    total = len(texts)
    done = 0

    def _extract(item) -> Tuple[int, ExtractionResult]:
        idx, text = item
        return idx, extract_from_text(text, paper_title)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_extract, item) for item in texts]
        for fut in as_completed(futures):
            try:
                idx, result = fut.result()
                for e in result.entities:
                    name = e.name.strip()
                    entities.add((e.type, name))
                    mentions.append({"name": name, "type": e.type, "chunk_index": idx})
                for r in result.relations:
                    relations.add((r.head.strip(), r.relation, r.tail.strip()))
                for p in result.performances:
                    performances.append({
                        "method": p.method.strip(),
                        "metric": p.metric.strip(),
                        "value": p.value.strip(),
                        "condition": p.condition.strip(),
                        "dataset": p.dataset.strip(),
                        "chunk_index": idx,
                    })
            except Exception as e:
                print(f"    [块失败] {e}")
            done += 1
            if done % 10 == 0 or done == total:
                print(f"    {done}/{total} chunks")

    return entities, relations, mentions, performances


def _dedup_dicts(items: list) -> list:
    """按字典内容去重（保证缓存稳定、无重复 mention/performance）。"""
    seen = set()
    out = []
    for d in items:
        key = tuple(sorted(d.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def serialize(entities: Set, relations: Set, mentions: list, performances: list) -> dict:
    """把集合结构转成可 JSON 序列化的字典，排序保证缓存文件稳定可读。"""
    mentions = sorted(
        _dedup_dicts(mentions),
        key=lambda m: (m["chunk_index"], m["type"], m["name"]),
    )
    performances = sorted(
        _dedup_dicts(performances),
        key=lambda p: (p["chunk_index"], p["method"], p["metric"]),
    )
    return {
        "entities": sorted(
            [{"type": t, "name": n} for t, n in entities],
            key=lambda x: (x["type"], x["name"]),
        ),
        "relations": sorted(
            [{"head": h, "relation": r, "tail": t} for h, r, t in relations],
            key=lambda x: (x["head"], x["relation"], x["tail"]),
        ),
        "performances": performances,
        "mentions": mentions,
    }


# ========== 6. 主流程：遍历目录 + 缓存 ==========

def process_all(limit: int | None = None) -> None:
    cache = load_cache()
    md_files = sorted(DATA_DIR.glob("*.md"))
    if limit is not None:
        md_files = md_files[:limit]

    total_files = len(md_files)
    for idx, md_file in enumerate(md_files, 1):
        raw_text = md_file.read_text(encoding="utf-8")
        key = md_file.name  # 用文件名做缓存 key
        current_hash = content_hash(raw_text)

        # 命中缓存且内容未变且 schema 一致，跳过
        entry = cache.get(key)
        if (
            entry
            and entry.get("content_hash") == current_hash
            and entry.get("schema_version") == SCHEMA_VERSION
        ):
            print(f"[缓存命中 {idx}/{total_files}] {md_file.name}")
            continue

        paper_title = md_file.stem.replace("_", " ")
        print(f"[抽取中 {idx}/{total_files}] {md_file.name} ...")

        try:
            chunks = split_markdown(raw_text)
            entities, relations, mentions, performances = extract_paper(chunks, paper_title)
        except Exception as e:
            print(f"[失败] {md_file.name}: {e}")
            continue  # 不写缓存，下次重跑会重试

        cache[key] = {
            "content_hash": current_hash,
            "title": paper_title,
            "schema_version": SCHEMA_VERSION,
            **serialize(entities, relations, mentions, performances),
        }
        save_cache(cache)  # 每篇抽完立即落盘，中断不丢
        print(
            f"[完成 {idx}/{total_files}] {md_file.name}：{len(entities)} 实体, "
            f"{len(relations)} 关系, {len(performances)} 性能, {len(mentions)} 溯源"
        )

    print(f"\n完成：共 {len(md_files)} 篇，缓存条目 {len(cache)} 条")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="阶段2：从 markdown 抽取实体/关系/性能结论")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（冒烟测试用）")
    args = parser.parse_args()
    process_all(limit=args.limit)



