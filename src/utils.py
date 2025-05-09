import re
import os
import yaml
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from src.hybrid_retriever import HybridRetriever

def create_AzureOpenAIEmbedding_3_large():
    return AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        deployment_name="text-embedding-3-large",
        api_key=os.environ.get("API_KEY_EMB"),
        azure_endpoint=os.environ.get("AZURE_ENDPOINT_EMB"),
        api_version="2024-02-01",
        dimensions=1024,
    )

def remove_redundancy_section(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    start_idx = None
    reference_start_idx = None

    # 查找“## ABSTRACT”、“## 1”和“## References”的索引
    for i, line in enumerate(lines):
        if "## ABSTRACT" in line or "## 1" in line:
            start_idx = i
            break

    for i, line in enumerate(lines):
        if "## References" in line:
            reference_start_idx = i
            break

    # 如果找到“## References”，删除之后的所有行
    if reference_start_idx is not None:
        lines = lines[start_idx:reference_start_idx] if start_idx is not None else lines[:reference_start_idx]
    elif start_idx is not None:
        lines = lines[start_idx:]

    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print(f'文件 "{file_path}" 中的指定部分已删除。')

def parse_steps(example_string):
    # Extract content inside the first pair of triple backticks
    content_match = re.search(r'```(.*)```', example_string, re.DOTALL)
    if content_match:
        example_string = content_match.group(1).strip()
    
    # Regular expression to match step instructions
    step_regex = re.compile(r"Step (\d+):\s*(?:#([^#]+)#)?\s*(.*?)(?=Step \d+:|$)", re.DOTALL)

    steps_list = []
    for match in step_regex.finditer(example_string):
        step_number = int(match.group(1))
        step_name = match.group(2).strip() if match.group(2) else ""
        step_instruction = match.group(3).strip()
        
        step_dict = {
            "step_number": step_number,
            "step_name": step_name,
            "step_instruction": step_instruction
        }
        steps_list.append(step_dict)
    
    return steps_list

def load_config_from_yaml(file_path: str) -> dict:
    """从YAML文件中加载配置"""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_vector_retriever(nodes, db_path, embed_model, top_k):
    if not os.path.exists(db_path):
        vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
        vector_index.storage_context.persist(persist_dir=db_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=db_path), 
            embed_model=embed_model
            )

    return vector_index.as_retriever(similarity_top_k=top_k)

def create_bm25_retriever(nodes, top_k):
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )

def create_hybrid_retriever(vector_retriever, bm25_retriever, top_k):
    if vector_retriever and bm25_retriever:
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=top_k
        )

