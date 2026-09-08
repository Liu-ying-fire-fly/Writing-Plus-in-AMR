# Writing_Plus

[English](README_EN.md) | 中文

自动调制识别（AMR）论文写作辅助系统：从本地论文库构建「向量索引 + 知识图谱」，实现
「向量召回 → 图遍历扩展 → cross-encoder 重排 → 两阶段精炼生成」的 RAG 写作流水线，
最终输出带 `[1][2]` 编号引用的论文初稿。

## 流水线

| 阶段 | 脚本 | 作用 |
|------|------|------|
| 索引 | `update_index.py` | 对 `markdown_data/` 分块 + 向量化，增量构建 FAISS 索引 |
| 抽取 | `02_relation_extraction.py` | LLM 抽取实体 / 关系 / 性能结论，并记录 entity↔chunk 归属 |
| 建图 | `03_graph_build.py` | 写入 Neo4j（Paper / Entity / Chunk / PerformanceResult 节点 + 语义边） |
| 检索 | `04_graph_retrieval.py` | 向量召回 + 图遍历扩展（合并去重） |
| 生成 | `05_refine_generate.py` | cross-encoder 重排 + 摘要压缩 + 精炼生成 |

另有 `01_langchain_example.py` 为早期示例，`print_content.py` 用于查看向量块内容。

## 目录结构

```
.
├── 01_*.py / 02_*.py / ... / 05_*.py   # 流水线各阶段
├── update_index.py                     # 向量索引
├── embeddings.py                       # Qwen3 嵌入封装
├── config.py / config.json             # 统一配置（点号键读取）
├── abbreviations.json                  # 缩写 → 全称规范表
├── markdown_data/                      # 论文 markdown（自行放入，见其 README）
├── model/                              # 嵌入模型权重（自行下载，见其 README）
└── paper_vertical/                     # FAISS 索引（运行 update_index.py 生成）
```

## 安装 Docker（Linux）

### 1. 安装 Docker

在终端执行一键安装脚本：

```bash
sudo curl -fsSL https://github.com/tech-shrimp/docker_installer/releases/download/latest/linux.sh | bash -s docker --mirror Aliyun
```

备用安装命令：

```bash
sudo curl -fsSL https://gitee.com/tech-shrimp/docker_installer/releases/download/latest/linux.sh | bash -s docker --mirror Aliyun
```

### 2. 配置 Docker 镜像源

编辑配置文件：

```bash
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://docker.m.daocloud.io",
                       "https://docker.1panel.live",
                       "https://hub.rat.dev"
                       ]
}
EOF
```

然后重启 Docker 服务：

```bash
sudo systemctl restart docker
```

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 准备数据与模型：
   - 将论文 `.md` 放入 `markdown_data/`（见 `markdown_data/README.md`）；
   - 下载嵌入模型到 `model/`（见 `model/README.md`）。
3. 配置环境变量（密钥不走配置文件）：
   ```bash
   export DEEPSEEK_API_KEY=...
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=...
   ```
4. 启动 Neo4j（Docker，默认 `neo4j` 数据库）：
   ```bash
   docker run -d --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/<YOUR_PASSWORD> \
     -v neo4j_data:/data \
     -v neo4j_logs:/logs \
     neo4j:5-community
   ```

   > `<YOUR_PASSWORD>` 需与上一步 `NEO4J_PASSWORD` 保持一致。

   然后依次运行流水线：
   ```bash
   python3 update_index.py
   python3 02_relation_extraction.py
   python3 03_graph_build.py --reset
   python3 04_graph_retrieval.py --query "what is AMR"
   python3 05_refine_generate.py --query "小样本识别的最好性能是哪篇论文？"
   ```

## 配置

所有可调参数集中在 `config.json`，通过 `config.py` 的点号键读取（如 `chunking.chunk_size`、`embedding.model_path`）。

> 注意：修改 `chunking` 相关参数会改变分块结果，需重新运行 `update_index.py`、`02`、`03` 全量重建。

## 安全

API key 与数据库密码一律通过环境变量传入，不写入任何配置文件或代码。

## 联系方式

huangjt@cjlu.edu.cn


