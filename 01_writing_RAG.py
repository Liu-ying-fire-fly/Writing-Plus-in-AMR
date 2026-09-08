import os

import config

# hugging face镜像设置，如果国内环境无法使用启用该设置
os.environ['HF_ENDPOINT'] = config.get("hf_endpoint", "https://hf-mirror.com")
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from modelscope import snapshot_download
from embeddings import Qwen3Embeddings

load_dotenv()

# # 中文嵌入模型
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
  
# Qwen 嵌入模型
embeddings = Qwen3Embeddings(
    model_path=str(config.path("embedding.model_path")),   # 对应 model_name
    device=config.get("embedding.device", "cuda"),          # 对应 model_kwargs['device']
    max_length=config.get("embedding.max_length", 2048),    # 最大 token 数
    batch_size=config.get("embedding.batch_size", 16),      # 批处理大小
)

# 构建/加载向量存储（持久化到本地，已存在则直接加载）
faiss_index_path = str(config.path("paths.faiss_index"))

vectorstore = FAISS.load_local(
    faiss_index_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。使用中文进行回答。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
# llm = ChatOpenAI(
#     model="glm-4.7-flash-free",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://aihubmix.com/v1"
# )

llm = ChatOpenAI(
    model=config.get("deepseek.model"),
    temperature=config.get("rag_demo.temperature", 0.7),
    max_tokens=config.get("rag_demo.max_tokens", 4096),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=config.get("deepseek.base_url")
)

# 用户查询
question = "什么是自动调制识别AMR"
print(question)

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=5)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
## 打印查找到的相关内容
# print(docs_content)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)


