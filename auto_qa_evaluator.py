import argparse
import os
import json
from typing import Dict
from src.utils import (
    load_config_from_yaml, 
    create_AzureOpenAIEmbedding_3_large,
    create_hybrid_retriever,
    create_bm25_retriever,
    create_vector_retriever
    )
from src.nodes_generators import NodesGenerator
from src.evaluators import Evaluator
from src.generators import OpenAIGenerator, AI01Generator, DeepSeekGenerator

def create_model_map(model_list) -> Dict:
    model_map = {}

    if 'gpt-4o' in model_list:
        llm_gpt4o = OpenAIGenerator()
        model_map[llm_gpt4o] = 'gpt-4o'
    
    if 'yi-large' in model_list:
        llm_01 = AI01Generator()
        model_map[llm_01] = 'yi-large'
    
    if 'deepseek' in model_list:
        llm_deepseek = DeepSeekGenerator()
        model_map[llm_deepseek] = 'deepseek-chat'

    return model_map

def create_retriever_map(retriever_list, top_k, chunk_size, chunk_overlap, directory_path) -> Dict:
    retriever_map = {}
    
    nodes_generator = NodesGenerator()
    nodes = nodes_generator.generate_nodes(
        dataset_path=directory_path, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    embed_model = create_AzureOpenAIEmbedding_3_large()
    db_path = f"./db/{chunk_size}_{chunk_overlap}"

    for idx, node in enumerate(nodes, start=1):
        node.metadata["chunk_idx"] = idx

    vector_retriever, bm25_retriever = None, None
    if 'vector_retriever' in retriever_list:
        vector_retriever = create_vector_retriever(
            nodes, db_path, embed_model, top_k
            )
        retriever_map[vector_retriever] = 'vector_retriever'

    if 'bm25_retriever' in retriever_list:
        bm25_retriever = create_bm25_retriever(nodes, top_k)
        retriever_map[bm25_retriever] = 'bm25_retriever'

    if 'hybrid_retriever' in retriever_list:
        if vector_retriever is None:
            vector_retriever = create_vector_retriever(
                nodes, db_path, embed_model, top_k
                )
        
        if bm25_retriever is None:
            bm25_retriever = create_bm25_retriever(nodes, top_k)
        
        hybrid_retriever = create_hybrid_retriever(vector_retriever, bm25_retriever, top_k)
        retriever_map[hybrid_retriever] = 'hybrid_retriever'

    return retriever_map

def main():
    parser = argparse.ArgumentParser(description="Run qa_evaluator with specified parameters")
    parser.add_argument("--config_file", type=str, default='./config/evaluator.yaml', help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_config_from_yaml(args.config_file)

    print(f"QA input file: {config['QA_input_file']}")
    print(f"original text input folder: {config['text_input_folder']}")
    print(f"Output folder: {config['output_folder']}")
    print(f"Retrievers: {config['retrievers']}")
    print(f"Top_k: {config['top_k']}")
    print(f"Nodes cutting: chunk_size: {config['chunk_size']}, chunk_overlap: {config['chunk_overlap']}")
    print(f"Models: {config['models']}")
    
    model_map = create_model_map(config['models'])

    retriever_map = create_retriever_map(
        retriever_list=config['retrievers'],
        top_k=config['top_k'], 
        chunk_size=config['chunk_size'], 
        chunk_overlap=config['chunk_overlap'], 
        directory_path=config['text_input_folder']
        )
    
  
    with open(config['QA_input_file'], 'r', encoding='utf-8') as f:
        queries = json.load(f)
       
    evaluator = Evaluator(OpenAIGenerator())
    output_folder = config['output_folder']

    for model_name in model_map.values():
        evaluation_res_folder = os.path.join(output_folder, "evaluation_res", model_name)
        os.makedirs(evaluation_res_folder, exist_ok=True)

    print("------Answer directly without any context------")
    for llm_to_pre in model_map.keys():

        final_res, total_score, score_ratio = evaluator.calculate_score_without_context(
            queries=queries, llm_to_pre=llm_to_pre, model_map=model_map
            )
        score_store_file = f"{config['output_folder']}/evaluation_res/{model_map[llm_to_pre]}/scores_without_context_pre_by_{model_map[llm_to_pre]}.json"

        data_to_store = {
            "total_score": total_score,
            "score_ratio": score_ratio,
            "results": final_res
        }

        with open(score_store_file, 'w') as f:
            json.dump(data_to_store, f, ensure_ascii=False, indent=4)

    print("------Answer with the retrieved context------")
    for llm_to_pre in model_map.keys():
        for retriever in retriever_map.keys():

            final_res, total_score, score_ratio, total_reference, reference_ratio = evaluator.calculate_score_with_context(
                queries=queries, retriever=retriever, top_k=config['top_k'], llm_to_pre=llm_to_pre, model_map=model_map, retriever_map=retriever_map
                )
            score_store_file = f"{config['output_folder']}/evaluation_res/{model_map[llm_to_pre]}/scores_with_context_pre_by_{model_map[llm_to_pre]}_retrieve_by_{retriever_map[retriever]}.json"

            data_to_store = {
                "total_score": total_score,
                "score_ratio": score_ratio,
                "total_reference": total_reference,
                "reference_ratio": reference_ratio,
                "results": final_res
            }

            with open(score_store_file, 'w') as f:
                json.dump(data_to_store, f, ensure_ascii=False, indent=4)

    print("------Answer with real context------")
    for llm_to_pre in model_map.keys():

        final_res, total_score, score_ratio = evaluator.calculate_score_with_true_context(queries=queries, llm_to_pre=llm_to_pre, model_map=model_map)
        score_store_file = f"{config['output_folder']}/evaluation_res/{model_map[llm_to_pre]}/scores_with_true_context_pre_by_{model_map[llm_to_pre]}.json"

        data_to_store = {
            "total_score": total_score,
            "score_ratio": score_ratio,
            "results": final_res
        }

        with open(score_store_file, 'w') as f:
            json.dump(data_to_store, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    main()
