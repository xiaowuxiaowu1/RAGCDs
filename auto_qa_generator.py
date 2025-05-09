import argparse
import json
import os
import yaml
from tqdm import tqdm
from src.utils import (
    remove_redundancy_section
    )
from src.evolers import Evolver, INITIAL_EVOLVE_METHOD
from src.analyzers import Analyzer
from src.generators import OpenAIGenerator, AI01Generator
from src.optimizers import Optimizer
from src.nodes_generators import NodesGenerator
from src.judges import Judge
from src.extractors import Extractor
from src.greedy_groupers import GreedyGrouper
from src.qa_generators import QAGenerator

def main():
    parser = argparse.ArgumentParser(description="Run qa_generator with specified parameters")
    parser.add_argument("--config_file", type=str, default='./config/generator.yaml', help="Path to the YAML config file")
    args = parser.parse_args()
 
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
      
    print(f"Input folder: {config['input_folder']}")
    print(f"Output folder: {config['output_folder']}")
    print(f"Iter time: {config['iter_time']}")
    print(f"Generator: {config['generator']}")

    # 主生成器 用于较为困难的任务
    major_generator = (
        OpenAIGenerator() 
        if config['generator'] == 'gpt-4o' 
        else AI01Generator()
    )

    # 次生成器 出于成本考虑，用于较为简单的任务
    second_generator = (
        AI01Generator()
    )

    evolver = Evolver(major_generator) # 演化器 用于instruction的生成
    analyzer = Analyzer(major_generator) # 分析器 用于prompt的评估反馈
    optimizer = Optimizer(major_generator) # 优化器 用于prompt的优化
    nodes_generator = NodesGenerator() # nodes生成器 把处理后的论文处理为nodes
    judge = Judge(second_generator) # 判断器 判断nodes是否适合于RAG任务
    extractor = Extractor(second_generator) # 提取器 从nodes中提取出claim topic 
    greedy_grouper = GreedyGrouper() # 贪心聚合器 把claim相似的nodes合并为一个claim set
    qa_generator = QAGenerator(major_generator) # qa生成器 根据claim set 生成qa对

    # 去除数据集中不需要的数据
    for file_name in os.listdir(config['input_folder']):
        if file_name.endswith('.txt'):
            remove_redundancy_section(os.path.join(config['input_folder'], file_name))

    nodes = nodes_generator.generate_nodes(dataset_path=config['input_folder'], chunk_size=200, chunk_overlap=20)
    reference_chunks, failed_chunks = judge.judge(nodes)
    
    with open(os.path.join(config['output_folder'], "reference_chunks.json"), 'w') as json_file:
        json.dump(reference_chunks, json_file, ensure_ascii=False, indent=4)
    with open(os.path.join(config['output_folder'], "failed_chunks.json"), 'w') as json_file:
        json.dump(failed_chunks, json_file, ensure_ascii=False, indent=4)
    print("Successfully obtained reference_chunks!")

    claim_collection = extractor.extract(reference_chunks)

    with open(os.path.join(config['output_folder'], "claim_collection.json"), 'w') as json_file:
        json.dump(claim_collection, json_file, ensure_ascii=False, indent=4)
    print("Successfully obtained claim_collection!")

    claim_sets = greedy_grouper.build_claim_set(claim_collection)

    with open(os.path.join(config['output_folder'], "claim_sets.json"), 'w') as json_file:
        json.dump(claim_sets, json_file, ensure_ascii=False, indent=4)
    print("Successfully obtained claim_sets!")

    test_queries = qa_generator.test_queries_generate(claim_sets, config['iter_time'])

    prompt = INITIAL_EVOLVE_METHOD 

    for i in tqdm(range(1, config['iter_time'] + 1), desc="Instruction iteration Progress"):
        print(i)
        query = test_queries[i-1]['raw_query']
        context = test_queries[i-1]['Claims'][0]['text']
        evolved_instructions, current_method = evolver.evolve(query=query, context=context, evolving_method=prompt)
        feedback = analyzer.analyze(query=query, context=context, evolved_instructions=evolved_instructions)
        optimized_instruction = optimizer.optimize(feedback=feedback, current_method=current_method)
        prompt = optimized_instruction  
        file_name = f"optimized_prompt_iter{i}.txt"
        file_name = os.path.join(config['output_folder'], file_name)
        with open(file_name, 'w') as file:
            file.write(prompt)

    qa_store_file = os.path.join(config['output_folder'], "queries.json")
    qa_generator.processing_loop(claim_sets, prompt, qa_store_file, 10)
    

if __name__ == "__main__":
    main()

