"""阶段3：把阶段2的抽取结果写入 Neo4j 图数据库。

节点：
- Paper  {filename, title}                 —— 唯一键 filename
- Entity {name, type}                      —— 唯一键 (name, type)
- Chunk  {chunk_id, filename, index, text} —— 唯一键 chunk_id（= 文件名::序号，与阶段1 FAISS 对齐）
- PerformanceResult {result_id, metric, value, value_num, condition, dataset} —— 唯一键 result_id

语义边（阶段2 的关系类型）：
- Paper-[USES|TARGETS|ADDRESSES]->Entity
- Entity-[EVALUATED_ON|APPLIES_TO|INVOLVES]->Entity

性能边：
- Entity(Method)-[:REPORTED]->PerformanceResult
- PerformanceResult-[:OF_METRIC]->Entity(Metric)
- PerformanceResult-[:ON_DATASET]->Entity(Dataset)
- Paper-[:HAS_RESULT]->PerformanceResult

溯源边：
- Chunk-[PART_OF]->Paper
- Entity-[MENTIONED_IN]->Chunk

entity↔chunk 映射直接读取阶段2 抽取时记录的 mentions（不再子串重匹配）；
实体名写库前经 abbreviations.json 把缩写（ACC/CNN/…）合并到全称。
"""

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config

load_dotenv()


# ========== 1. 路径与连接配置 ==========

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = config.path("paths.cache_path")
MARKDOWN_DIR = config.path("paths.data_dir")

# 与阶段1/2 完全一致，保证 chunk id 与切分结果对齐
CHUNK_SIZE = config.get("chunking.chunk_size", 1200)
CHUNK_OVERLAP = config.get("chunking.chunk_overlap", 300)
HEADERS_TO_SPLIT_ON = config.headers()

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=True,
)

sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def header_prefix(metadata: dict) -> str:
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
    """与 02_relation_extraction.split_markdown / update_index.py 保持一致。"""
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


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


# ========== 2. 实体规范化与索引 ==========

def norm_key(s: str) -> str:
    """实体名与 chunk 文本的规范化键：NFKC + 连字符/下划线转空格 + 折叠大小写。"""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


ABBREV_PATH = config.path("paths.abbreviations")


def load_abbreviations() -> dict:
    if ABBREV_PATH.exists():
        return json.loads(ABBREV_PATH.read_text(encoding="utf-8"))
    return {}


# 缩写 -> 全称（键已规范化）：写库前把 ACC/CNN 等统一合并到全称实体
ABBREVIATIONS = {norm_key(k): v for k, v in load_abbreviations().items()}


def apply_abbreviation(name: str) -> str:
    """命中缩写表则返回全称，否则原样返回。"""
    full = ABBREVIATIONS.get(norm_key(name))
    return full if full else name


def _display_score(name: str, count: int):
    """选规范展示名：次数优先，其次偏好「首字母大写」的自然写法，最后按字典序兜底。"""
    has_upper = any(c.isupper() for c in name)
    has_lower = any(c.islower() for c in name)
    proper = 1 if (has_upper and has_lower) else 0
    return (count, proper, name)


def build_entity_index(cache: dict):
    """统计全库实体名变体，返回 {(norm_key, type): canonical_name} 与实体节点列表。"""
    freq = defaultdict(Counter)
    for entry in cache.values():
        for e in entry.get("entities", []):
            name = apply_abbreviation(e["name"].strip())
            if not name:
                continue
            freq[(norm_key(name), e["type"])][name] += 1

    index = {}
    entity_nodes = []
    for (key, type_), counter in freq.items():
        canonical = max(counter.items(), key=lambda kv: _display_score(kv[0], kv[1]))[0]
        index[(key, type_)] = canonical
        entity_nodes.append({"type": type_, "name": canonical})
    return index, entity_nodes


# 关系类型 -> (head 期望类型, tail 期望类型)；"Paper" 表示该端点指向当前论文
RELATION_SCHEMA = {
    "USES":         ("Paper", "Method"),
    "TARGETS":      ("Paper", "Task"),
    "ADDRESSES":    ("Paper", "ProblemScenario"),
    "EVALUATED_ON": ("Method", "Dataset"),
    "APPLIES_TO":   ("Method", "ProblemScenario"),
    "INVOLVES":     ("ProblemScenario", "Task"),
}
PAPER_HEAD_RELS = ["USES", "TARGETS", "ADDRESSES"]
ENTITY_HEAD_RELS = ["EVALUATED_ON", "APPLIES_TO", "INVOLVES"]


def resolve_entity(name: str, type_: str, index: dict, entity_nodes: list) -> str:
    """按 (norm_key, type) 解析实体名；relation 里出现 entities 列表之外的名称时兜底创建。"""
    name = apply_abbreviation(name.strip() or name)
    key = (norm_key(name), type_)
    if key in index:
        return index[key]
    canonical = name
    index[key] = canonical
    entity_nodes.append({"type": type_, "name": canonical})
    return canonical


# ========== 3. 内存态构建：节点 + 边 ==========

def _parse_value(value: str):
    """从数值字符串里提取首个数字，无法解析返回 None。"""
    m = re.search(r"-?\d+(?:\.\d+)?", value or "")
    return float(m.group()) if m else None


def build_all(cache: dict):
    """把缓存一次性展开为可写入的节点/边列表。"""
    index, entity_nodes = build_entity_index(cache)

    papers = []
    chunks = []
    mentions = []
    paper_edges = {rt: [] for rt in PAPER_HEAD_RELS}
    entity_edges = {rt: [] for rt in ENTITY_HEAD_RELS}
    result_nodes = []
    result_edges = {"REPORTED": [], "OF_METRIC": [], "ON_DATASET": [], "HAS_RESULT": []}

    for filename in sorted(cache):
        entry = cache[filename]
        title = entry.get("title") or filename
        papers.append({"filename": filename, "title": title})

        # 重建 chunk（与阶段1/2 相同规则，id = 文件名::序号）
        md_file = MARKDOWN_DIR / filename
        if md_file.exists():
            docs = split_markdown(md_file.read_text(encoding="utf-8"))
        else:
            docs = []
        texts = [d.page_content for d in docs]
        for i, text in enumerate(texts):
            chunks.append({
                "chunk_id": f"{filename}::{i}",
                "filename": filename,
                "index": i,
                "text": text,
                "title": title,
            })

        # entity -> chunk 溯源：直接读阶段2抽取时记录的 mentions
        for m in entry.get("mentions", []):
            canonical = resolve_entity(m["name"], m["type"], index, entity_nodes)
            mentions.append({
                "name": canonical,
                "type": m["type"],
                "chunk_id": f"{filename}::{m['chunk_index']}",
            })

        # 性能结论 -> PerformanceResult 节点 + REPORTED/OF_METRIC/ON_DATASET/HAS_RESULT 边
        for p in entry.get("performances", []):
            value = (p.get("value") or "").strip()
            if not value:
                continue
            method_name = resolve_entity(p.get("method") or "", "Method", index, entity_nodes)
            metric_name = resolve_entity(p.get("metric") or "", "Metric", index, entity_nodes)
            condition = (p.get("condition") or "").strip()
            dataset_raw = (p.get("dataset") or "").strip()
            result_id = hashlib.md5(
                f"{filename}|{method_name}|{metric_name}|{value}|{condition}|{dataset_raw}".encode("utf-8")
            ).hexdigest()
            result_nodes.append({
                "result_id": result_id,
                "metric": metric_name,
                "value": value,
                "value_num": _parse_value(value),
                "condition": condition,
                "dataset": dataset_raw,
            })
            result_edges["REPORTED"].append({
                "head_name": method_name,
                "head_type": "Method",
                "result_id": result_id,
            })
            result_edges["OF_METRIC"].append({
                "result_id": result_id,
                "tail_name": metric_name,
                "tail_type": "Metric",
            })
            if dataset_raw:
                dataset_name = resolve_entity(dataset_raw, "Dataset", index, entity_nodes)
                result_edges["ON_DATASET"].append({
                    "result_id": result_id,
                    "tail_name": dataset_name,
                    "tail_type": "Dataset",
                })
            result_edges["HAS_RESULT"].append({
                "filename": filename,
                "result_id": result_id,
            })

        # 语义边：head 为 Paper 的直接指当前论文，其余按期望类型解析实体
        for r in entry.get("relations", []):
            rt = r["relation"]
            if rt not in RELATION_SCHEMA:
                continue
            head_type, tail_type = RELATION_SCHEMA[rt]
            if head_type == "Paper":
                paper_edges[rt].append({
                    "filename": filename,
                    "tail_name": resolve_entity(r["tail"], tail_type, index, entity_nodes),
                    "tail_type": tail_type,
                })
            else:
                entity_edges[rt].append({
                    "head_name": resolve_entity(r["head"], head_type, index, entity_nodes),
                    "head_type": head_type,
                    "tail_name": resolve_entity(r["tail"], tail_type, index, entity_nodes),
                    "tail_type": tail_type,
                })

    return papers, entity_nodes, chunks, mentions, paper_edges, entity_edges, result_nodes, result_edges


# ========== 4. 写库 ==========

def create_constraints(session):
    session.run("CREATE CONSTRAINT paper_filename IF NOT EXISTS FOR (p:Paper) REQUIRE p.filename IS UNIQUE")
    session.run("CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE")
    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
    session.run("CREATE CONSTRAINT perf_result_id IF NOT EXISTS FOR (pr:PerformanceResult) REQUIRE pr.result_id IS UNIQUE")


def write_papers(session, papers):
    session.run(
        "UNWIND $rows AS row MERGE (p:Paper {filename: row.filename}) SET p.title = row.title",
        rows=papers,
    )


def write_entities(session, entity_nodes):
    session.run(
        "UNWIND $rows AS row MERGE (e:Entity {name: row.name, type: row.type})",
        rows=entity_nodes,
    )


def write_chunks(session, chunks):
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Paper {filename: row.filename})
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.filename = row.filename, c.index = row.index, c.text = row.text
        MERGE (c)-[:PART_OF]->(p)
        """,
        rows=chunks,
    )


def write_mentions(session, mentions):
    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {name: row.name, type: row.type})
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """,
        rows=mentions,
    )


def write_performance_results(session, result_nodes, result_edges):
    session.run(
        """
        UNWIND $rows AS row
        MERGE (pr:PerformanceResult {result_id: row.result_id})
        SET pr.metric = row.metric, pr.value = row.value,
            pr.value_num = row.value_num, pr.condition = row.condition,
            pr.dataset = row.dataset
        """,
        rows=result_nodes,
    )
    if result_edges["REPORTED"]:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (m:Entity {name: row.head_name, type: row.head_type})
            MATCH (pr:PerformanceResult {result_id: row.result_id})
            MERGE (m)-[:REPORTED]->(pr)
            """,
            rows=result_edges["REPORTED"],
        )
    if result_edges["OF_METRIC"]:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (pr:PerformanceResult {result_id: row.result_id})
            MATCH (t:Entity {name: row.tail_name, type: row.tail_type})
            MERGE (pr)-[:OF_METRIC]->(t)
            """,
            rows=result_edges["OF_METRIC"],
        )
    if result_edges["ON_DATASET"]:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (pr:PerformanceResult {result_id: row.result_id})
            MATCH (t:Entity {name: row.tail_name, type: row.tail_type})
            MERGE (pr)-[:ON_DATASET]->(t)
            """,
            rows=result_edges["ON_DATASET"],
        )
    if result_edges["HAS_RESULT"]:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (p:Paper {filename: row.filename})
            MATCH (pr:PerformanceResult {result_id: row.result_id})
            MERGE (p)-[:HAS_RESULT]->(pr)
            """,
            rows=result_edges["HAS_RESULT"],
        )


def write_semantic_edges(session, paper_edges, entity_edges):
    for rt, rows in paper_edges.items():
        if not rows:
            continue
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (p:Paper {{filename: row.filename}})
            MATCH (e:Entity {{name: row.tail_name, type: row.tail_type}})
            MERGE (p)-[:{rt}]->(e)
            """,
            rows=rows,
        )
    for rt, rows in entity_edges.items():
        if not rows:
            continue
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (h:Entity {{name: row.head_name, type: row.head_type}})
            MATCH (t:Entity {{name: row.tail_name, type: row.tail_type}})
            MERGE (h)-[:{rt}]->(t)
            """,
            rows=rows,
        )


def print_summary(session):
    print("\n节点统计：")
    for rec in session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"):
        print(f"  {rec['label']:12s} {rec['cnt']}")
    print("关系统计：")
    for rec in session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY cnt DESC"):
        print(f"  {rec['t']:16s} {rec['cnt']}")


# ========== 5. 主流程 ==========

def main():
    parser = argparse.ArgumentParser(description="把阶段2抽取结果写入 Neo4j")
    parser.add_argument("--reset", action="store_true", help="清空图数据库后重建")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", config.get("neo4j.uri", "bolt://localhost:7687")))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", config.get("neo4j.user", "neo4j")))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", config.get("neo4j.database", "neo4j")))
    args = parser.parse_args()

    if not args.password:
        print("未提供 NEO4J_PASSWORD，请通过环境变量或 --password 传入。")
        return

    cache = load_cache()
    if not cache:
        print("extraction_cache.json 为空或不存在，请先运行阶段2。")
        return

    papers, entity_nodes, chunks, mentions, paper_edges, entity_edges, result_nodes, result_edges = build_all(cache)
    total_relations = sum(len(v) for v in paper_edges.values()) + sum(len(v) for v in entity_edges.values())
    print(
        f"构建完成：{len(papers)} 篇论文, {len(entity_nodes)} 实体, "
        f"{len(chunks)} 块, {len(mentions)} 条 MENTIONED_IN, {total_relations} 条语义边, "
        f"{len(result_nodes)} 条性能结果"
    )

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    print(f"已连接 {args.uri}")

    with driver.session(database=args.database) as session:
        if args.reset:
            session.run("MATCH (n) DETACH DELETE n")
            print("已清空图数据库。")
        create_constraints(session)
        write_papers(session, papers)
        write_entities(session, entity_nodes)
        write_chunks(session, chunks)
        write_mentions(session, mentions)
        write_performance_results(session, result_nodes, result_edges)
        write_semantic_edges(session, paper_edges, entity_edges)
        print_summary(session)

    driver.close()
    print("写入完成。")


if __name__ == "__main__":
    main()





# docker run -d --name neo4j \
#   -p 7474:7474 -p 7687:7687 \
#   -e NEO4J_AUTH=neo4j/<YOUR_PASSWORD> \
#   -v neo4j_data:/data \
#   -v neo4j_logs:/logs \
#   neo4j:5-community

# export NEO4J_PASSWORD=<YOUR_PASSWORD>
# python3 03_graph_build.py --reset

