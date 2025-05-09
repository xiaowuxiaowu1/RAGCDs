import re
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from llama_index.core.retrievers import (
    BaseRetriever
)
from .base_evaluator import BaseEvaluator
from src.generators.base_generator import BaseGenerator

EVALUATION_PROMPT  = (
    "### Task Description:"
    "I need your help evaluating a comparison between a large language model's (LLM) predicted answer and the reference (ground truth) answer."
    "Your task is to determine whether the key facts and important information (emphasized with # symbols in the reference answer) are present in the LLM's predicted answer."

    "Please analyze the data provided and make a judgment based on accuracy, precision, completeness, and overall trend."

    "### Instructions:"
    "Carefully compare the 'Predicted Answer' and the 'Reference Answer'."
    "Determine whether the information emphasized with # symbols in the 'Reference Answer' appears in the 'Predicted Answer'."
    "Note that, unless exact wording is critical to the meaning, do not overly focus on the precise wording; if the meaning is consistent, it can be considered a successful match."
    "Your final judgment should be based on whether the meaning and key facts in the 'Reference Answer' are reflected in the 'Predicted Answer'."

    "Score according to the following criteria:"

    "- **5**: The 'Predicted Answer' **fully** covers all the key facts, numbers, and technical details emphasized with # symbols in the 'Reference Answer', and is highly consistent with it."
    "Even if phrasing differs, there are no significant omissions or errors."

    "- **3**: The 'Predicted Answer' is generally aligned with the overall trend of the 'Reference Answer' and contains no contradictory information,"
    "but it is missing one key fact emphasized with #, which affects the completeness of the answer."

    "- **1**: The 'Predicted Answer' is generally aligned with the overall trend of the 'Reference Answer' and contains no contradictory information,"
    "but it is missing more than one key fact emphasized with #, impacting the completeness of the answer."

    "- **0**: The 'Predicted Answer' contains information that contradicts the reference answer."

    "### Input Data:"
    "\nQuery: {query}"

    "\nPredicted Answer: {predicted_answer}"

    "\nReference Answer: {groundtruth_answer}"

    "### Output Format:"
    "Please provide your final evaluation in the following format:"
    "\nExplanation: (How did you make your decision based on the predicted and reference answers?"
    "Why did you assign this score?"
    "Explain whether key facts are present, trends are aligned, and if important details are missing."
    "Keep it within 50 words.)"
    "\nScore: (Your final score: 5, 3, 1, or 0)"
)

PREDICTED_ANSWER_PROMPT = (
    "You are an expert in the field of materials"
    "Your task is to generate an accurate answer to the query based on the knowledge you already possess. "
    "The answer should address all aspects of the query as thoroughly as possible, but keep it within 50 words."
    "Here is the query, and you need to generate the answer without providing any additional content: "
    "Query: {query}"
    "Now, please generate your answer:"
)

GENERATE_ANSWER_PROMPT = (
    "### Task Description:"
    "Generate a **concise and accurate answer**, based on the query and context."
    "First, assess if the context is relevant:"
    "- If the context is relevant, rely on it to answer;"
    "- If partially relevant, select useful information to answer;"
    "- If the context is irrelevant, ignore it and use the model's knowledge."

    "The generated answer must meet the following criteria:"
    "1. **Relevance Assessment**:"
    "   - First, assess if the context is relevant. If not, answer using model knowledge."
    "   - If partially relevant, only use related information and combine it with model knowledge."
    "2. **Context Dependency**:"
    "   - If the context is relevant, the answer must rely on it without introducing external information."
    "   - If partially relevant, combine selected context with model knowledge."
    "3. **Conciseness**:"
    "   - The answer should be within 50 words, keeping it concise and accurately answering the query."
    "4. **Logical Consistency**:"
    "   - The answer must be clear, logically structured, and consistent with either the context or model knowledge."

    "### Output Requirements:"
    "- Output only the answer, not the query or context."
    "- If the context is relevant, answer based on it; if irrelevant, use model knowledge."
    "- Ensure the answer is concise, accurate, and directly answers the query, within 50 words."

    "Here is the query and the context.\n"
    "Query: {query}\n"  
    "Context: {context}\n"
)

class Evaluator(BaseEvaluator):
    def __init__(self, generator: BaseGenerator) -> None:
        self.eval_generator = generator
    
    def calculate_score_without_context(self, queries, llm_to_pre: BaseGenerator, model_map):
        final_res = []
        total_score = 0  # 记录总分
        max_score = len(queries) * 5  # 计算满分
        pred_generator = llm_to_pre
        for item in tqdm(queries, desc=f"Calculating scores for {model_map[llm_to_pre]}"):
            query_idx = item["query_idx"]
            query = item["optimized_query"]
            groundtruth_answer = item["groundtruth_answer"]

            # 生成预测答案
            predicted_answer = pred_generator.generate(PREDICTED_ANSWER_PROMPT.format(query=query))

            # 生成评估结果
            evaluation_response = self.eval_generator.generate(EVALUATION_PROMPT.format(query=query, predicted_answer=predicted_answer, groundtruth_answer=groundtruth_answer))

            try:
                explanation = re.search(r'Explanation:\s*(.*)', evaluation_response).group(1).strip()
                score = re.search(r'Score:\s*(\d+)', evaluation_response).group(1).strip()
                score = int(score)
            except Exception as e:
                print(f"Error during parsing evaluation_response: {e}")
                print("The response that need to be addressed:\n",evaluation_response)
                explanation = "Error extracting explanation"
                score = 0  # 如果出现错误，将得分设为0

            result = {
                "query_idx": query_idx,
                "query": query,
                "groundtruth_answer": groundtruth_answer,
                "predicted_answer": predicted_answer,
                "score": score,
                "explanation": explanation,
            }
            final_res.append(result)

            total_score += score

        # 计算并输出总分与满分比值
        score_ratio = total_score / max_score
        print(f"Score without context: {total_score}/{max_score} ({score_ratio:.2%})")

        return final_res, total_score, score_ratio
    
    def calculate_score_with_context(self, queries, retriever: BaseRetriever, top_k: int, llm_to_pre: BaseGenerator, model_map, retriever_map):
        final_res = []
        total_score = 0  # 记录总分
        max_score = len(queries) * 5  # 计算满分
        total_reference = 0 # 记录总的检索成功的上下文
        max_reference = 0 # 参考上下文的总数
        pred_generator = llm_to_pre


        for item in tqdm(queries, desc=f"Calculating scores for {model_map[llm_to_pre]} retrieved by {retriever_map[retriever]}"):
            reference_list = []
            reference_idx = []
            retrieved_list = []
            contexts = ""
            query_idx = item["query_idx"]
            query = item["optimized_query"]
            context_num = item["num_of_contexts"]

            max_reference += context_num

            nodes = retriever.retrieve(query)
            for index, node in enumerate(nodes):
                contexts += f"Context {index}: {node.node.get_content()}\n"
                retrieved_list.append(node.node.get_content())
            for cliam_list in item["Claims"]:
                reference_list.append(cliam_list["text"])
                reference_idx.append(cliam_list["chunk_idx"])

            groundtruth_answer = item["groundtruth_answer"]

            # 检索结果
            hits_at_3, hits_at_top_k, mrr, hits_count= self.calculate_metrics(retrieved_list, reference_list, top_k=top_k)
            total_reference += hits_count

            # 生成预测答案
            predicted_answer = pred_generator.generate(GENERATE_ANSWER_PROMPT.format(query=query, context=contexts))

            # 生成评估结果
            evaluation_response = self.eval_generator.generate(EVALUATION_PROMPT.format(query=query, predicted_answer=predicted_answer, groundtruth_answer=groundtruth_answer))

            try:
                explanation = re.search(r'Explanation:\s*(.*)', evaluation_response).group(1).strip()
                score = re.search(r'Score:\s*(\d+)', evaluation_response).group(1).strip()
                score = int(score)
            except Exception as e:
                print(f"Error during parsing evaluation_response: {e}")
                print("The response that need to be addressed:\n",evaluation_response)
                explanation = "Error extracting explanation"
                score = 0  # 如果出现错误，将得分设为0

            result = {
                "query_idx": query_idx,
                "query": query,
                "retrieved_metrics" :{
                    "reference_idx": reference_idx,
                    "retrieved_list": retrieved_list,
                    "hits_at_3": hits_at_3,
                    "hits_at_top_k": hits_at_top_k,
                    "mrr": mrr,
                    "hits_count": hits_count,
                    "num_of_contexts": context_num,
                },
                "groundtruth_answer": groundtruth_answer,
                "predicted_answer": predicted_answer,
                "score": score,
                "explanation": explanation,
            }
            final_res.append(result)

            total_score += score

        # 计算并输出总分与满分比值
        score_ratio = total_score / max_score
        reference_ratio = total_reference / max_reference
        print(f"Score retrieved context: {total_score}/{max_score} ({score_ratio:.2%})")
        print(f"Total retrieve contexts: {total_reference}/{max_reference} ({reference_ratio:.2%})")

        return final_res, total_score, score_ratio, total_reference, reference_ratio
    
    def calculate_score_with_true_context(self, queries, llm_to_pre: BaseGenerator, model_map):
        final_res = []
        total_score = 0  # 记录总分
        max_score = len(queries) * 5  # 计算满分
        pred_generator = llm_to_pre

        for item in tqdm(queries, desc=f"Calculating scores for {model_map[llm_to_pre]}"):
            reference_idx = []
            contexts = ""
            query_idx = item["query_idx"]
            query = item["optimized_query"]
            context_num = item["num_of_contexts"]

            for index, cliam_list in enumerate(item["Claims"]):
                reference_idx.append(cliam_list["chunk_idx"])
                contexts += f"Context {index}: {cliam_list['text']}\n"

            groundtruth_answer = item["groundtruth_answer"]

            # 生成预测答案
            predicted_answer = pred_generator.generate(GENERATE_ANSWER_PROMPT.format(query=query, context=contexts))

            # 生成评估结果
            evaluation_response = self.eval_generator.generate(EVALUATION_PROMPT.format(query=query, predicted_answer=predicted_answer, groundtruth_answer=groundtruth_answer))

            try:
                explanation = re.search(r'Explanation:\s*(.*)', evaluation_response).group(1).strip()
                score = re.search(r'Score:\s*(\d+)', evaluation_response).group(1).strip()
                score = int(score)
            except Exception as e:
                print(f"Error during parsing evaluation_response: {e}")
                print("The response that need to be addressed:\n",evaluation_response)
                explanation = "Error extracting explanation"
                score = 0  # 如果出现错误，将得分设为0

            result = {
                "query_idx": query_idx,
                "query": query,
                "retrieved_metrics" :{
                    "reference_idx": reference_idx,
                    "num_of_contexts": context_num,
                },
                "groundtruth_answer": groundtruth_answer,
                "predicted_answer": predicted_answer,
                "score": score,
                "explanation": explanation,
            }
            final_res.append(result)

            total_score += score

        # 计算并输出总分与满分比值
        score_ratio = total_score / max_score
        print(f"Total score: {total_score}/{max_score} ({score_ratio:.2%})")

        return final_res, total_score, score_ratio
    
    def calculate_retrieve_ratio(self, queries, retriever: BaseRetriever, top_k: int, ):
        total_reference = 0 # 记录总的检索成功的上下文
        max_reference = 0 # 参考上下文的总数


        for item in tqdm(queries, desc="Calculating scores"):
            reference_list = []
            reference_idx = []
            retrieved_list = []
            contexts = ""
            query = item["raw_query"]
            context_num = item["num_of_contexts"]

            max_reference += context_num

            nodes = retriever.retrieve(query)
            for index, node in enumerate(nodes):
                contexts += f"Context {index}: {node.node.get_content()}\n"
                retrieved_list.append(node.node.get_content())
            for cliam_list in item["Claims"]:
                reference_list.append(cliam_list["text"])
                reference_idx.append(cliam_list["chunk_idx"])


            # 检索结果
            hits_at_3, hits_at_top_k, mrr, hits_count= self.calculate_metrics(retrieved_list, reference_list, top_k=top_k)
            total_reference += hits_count

        
        reference_ratio = total_reference / max_reference
        print(f"Total retrieve contexts: {total_reference}/{max_reference} ({reference_ratio:.2%})")

        return total_reference, reference_ratio
        
    
    def split_into_sentences(self, text):
        """ 使用正则表达式将文本分割成句子，并移除空字符串 """

        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = sentence_endings.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]  # 去掉空字符串

    def partial_match(self, gold_item, retrieved_item):
        """ 判断gold_item和retrieved_item是否有两个连续的句子匹配，或单句匹配 """

        gold_sentences = self.split_into_sentences(gold_item)
        retrieved_sentences = self.split_into_sentences(retrieved_item)
        
        # 如果其中之一只有一个句子，单句匹配即可
        if len(gold_sentences) == 1 or len(retrieved_sentences) == 1:
            for gold_sentence in gold_sentences:
                for retrieved_sentence in retrieved_sentences:
                    if gold_sentence.strip() == retrieved_sentence.strip():
                        return True
            return False

        # 如果有多个句子，检查是否有连续的两个句子匹配
        for i in range(len(gold_sentences) - 1): 
            gold_pair = (gold_sentences[i].strip(), gold_sentences[i + 1].strip())
            
            for j in range(len(retrieved_sentences) - 1): 
                retrieved_pair = (retrieved_sentences[j].strip(), retrieved_sentences[j + 1].strip())
                
                if gold_pair == retrieved_pair:
                    return True

        return False
    
    def calculate_metrics(self, retrieved_list, reference_list, top_k=5):
        hits_at_top_k_flag = False
        hits_at_3_flag = False
        first_relevant_rank = None

        gold = [item.replace("\n", " ") for item in reference_list]
        retrieved = [item.replace("\n", " ") for item in retrieved_list]
        find_gold = []
        count = 0
        for rank, retrieved_item in enumerate(retrieved[:top_k], start=1):
            if any(self.partial_match(gold_item, retrieved_item) for gold_item in gold):
                if rank <= top_k:
                    hits_at_top_k_flag = True
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    if rank <= 3:
                        hits_at_3_flag = True
                    for gold_item in gold:
                        if self.partial_match(gold_item, retrieved_item) and gold_item not in find_gold:
                            count =  count + 1
                            find_gold.append(gold_item)

        return int(hits_at_3_flag), int(hits_at_top_k_flag), 1 / first_relevant_rank if first_relevant_rank else 0, count
    