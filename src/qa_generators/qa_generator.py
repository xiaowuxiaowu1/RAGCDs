import json
import re
from tqdm import tqdm
from src.utils import parse_steps
from .base_qa_generator import BaseQAGenerator
from src.generators import BaseGenerator

MULTIHOP_QUERY_PROMPT = (
    "Your task is to create a multi-hop query based on the provided contexts and topic."
    "The answer to the query must consist of core points from all the contexts."
    "**However, the query must not directly include the exact words of the core points or their synonyms, near-synonyms.**"
    "Make sure that if any context is missing, it would be impossible to accurately answer the query."
    "The query must be precise and specific, asking about processes, terms, numbers, or specific information."
    "Try to use 'what' questions as much as possible, but 'how' can also be used when necessary."
    "The query must be semantically clear, easy to understand, and should clearly point to the desired answer."
    "Avoid using vague or broad language."

    "\n\nFollow these steps to complete the task:"

    "1. **Identify the core points**: Read each context and its topic, and identify a core point from each context."
    "The core point can be a word, a number, or a phrase, and it must be specific to that context, not common knowledge."
    "These core points will form the final answer."
    "Try to link the core points from different contexts as much as possible."

    "2. **Generate the query**: Design a single, clear, and specific query that "
    "requires information from all contexts to be answered."
    "Make sure that if any one context is ignored, the query cannot be answered."
    "The query should focus on the synthesis of all core points."
    "**Do not include core points, their exact words, or their synonyms, near-synonyms in the query.**"
    "The query must not, in any form, hint or reveal the content of the answer."

    "3. **Provide the answer**: Based on the final multi-hop query and all contexts,"
    "provide a concise and accurate answer."
    "The answer must directly respond to the query, rely on all contexts, include core points from all contexts, "
    "and exclude any irrelevant information."
    "When core points appear in the answer, use # to emphasize them."

    "\n### Example ###\n"
    "#### Context 1:\n"
    "After the doping of rare-earth element Tb^3+, the crystal structure of perovskite aluminate underwent significant changes."
    "#### Context 2:\n"
    "These structural changes enhanced its luminescence performance."
    "#### Core Points:\n"
    "Context 1 Core Point: Doping with Tb^3+ caused crystal structure changes"
    "Context 2 Core Point: Enhanced luminescence performance"

    "#### STEP1: (Core Points)\n"
    "Tb^3+, luminescence performance"

    "#### STEP2: (Query)\n"
    "Which element's doping causes structural changes in perovskite aluminate and leads to performance improvement?"

    "#### STEP3: (Answer)\n"
    "Doping with #Tb^3+# causes changes in the crystal structure of perovskite aluminate, "
    "enhancing its #luminescence performance#."
    "# The emphasized terms must be the core points, and the number of core points must match the number of contexts."
    "The actual multi-hop query will rely on multiple contexts, and the answer will include core points from all contexts."

    "\n\nHere are the contexts:\n"
    "{contexts}"
    "\nBelow is the corresponding topic:\n"
    "{topic}"

    "\nPlease strictly follow the format below and replace the content after the colons with your response:"
    "\nSTEP1: [List the core points identified from each context]"
    "\nSTEP2: [Provide the final multi-hop query]"
    "\nSTEP3: [Provide the answer to the query, with core points emphasized using #]"
    "Ensure that the query can only be answered by combining all the contexts and not relying on any single context."
)

OPTIMIZE_MULTIHOP_QUERY_PROMPT = (
    "Your task is to review and optimize the generated multi-hop query and its corresponding answer based on the provided contexts."
    "The answer to the multi-hop query includes content emphasized with # symbols, "
    "indicating the core points included in the answer."
    "The core points come from multiple contexts and are also provided separately."
    "You need to follow these steps:"
    
    "Step 1. **Suggestions for modifying the multi-hop query**: "
    "How can the multi-hop query be modified to ensure the following:"
    "  - Ensure that all contexts are required to answer it. If any context is missing, "
    "the multi-hop query cannot be answered."
    "  - **Ensure the query does not include core points, their exact words, synonyms, or near-synonyms.** "
    "Core points should only appear in the answer."
    "  - Ensure the query is fluent and can be understood independently without requiring additional context "
    "(for example, do not reference 'Context 1,' 'Context 2,' etc.)."
    "  - The query should be precise and specific, asking about processes, terms, numbers, "
    "or specific information, avoiding vague or broad language."
    "  - The query must not hint at or reveal any part of the answer in any form."
    "  Provide your targeted suggestions for modification."
    
    "Step 2. **Modify the multi-hop query based on the suggestions**: "
    "Modify the multi-hop query to ensure it cannot be answered by relying on a single context."
    "  - Generate an optimized multi-hop query that evenly relies on all contexts and can be answered using the core points."
    "  - The query must be semantically clear, easy to understand, and clearly point to the required answer."
    
    "Step 3. **Suggestions for modifying the answer**: How can the answer be modified to ensure the following:"
    "  - Ensure the answer semantically and accurately addresses all aspects of the optimized multi-hop query from Step 2."
    "  - **Ensure the answer includes core points from all contexts and emphasizes them using the # symbol. "
    "The number of core points should match the number of contexts.**"
    "  - Ensure the answer is as concise as possible, ideally within 30 words. "
    "If the query cannot be fully answered within 30 words, the word count can be increased appropriately."
    "  Provide your targeted suggestions for modification."
    
    "Step 4. **Modify the answer based on the suggestions**: "
    "Modify the answer to precisely address the optimized multi-hop query from Step 2."
    "  - When core points appear in the answer, emphasize them using the # symbol."
    "  - For example: The tensile strength of the alloy remains at #500 MPa#, "
    "#corrosion resistance# has significantly improved, and #wear resistance# is enhanced."
    
    "\nHere is the previously generated multi-hop query:\n"
    "{multihop_query}"
    "\nHere is the answer to the multi-hop query:\n"
    "{answer}"
    "\nHere are the core points from the multi-hop query answer:\n"
    "{core_points}"
    "\nHere are the corresponding contexts:\n"
    "{contexts}"
    
    "\nPlease strictly follow the format below and replace the content after the colons with your actual responses:"
    "\nSTEP1: [Replace with your suggestions for modification]"
    "\nSTEP2: [Replace with the modified multi-hop query]"
    "\nSTEP3: [Replace with your suggestions for modification]"
    "\nSTEP4: [Replace with the modified answer]"
)

SINGLE_QUERY_PROMPT = (
    "### Task Description:\n"
    "Your task is to generate a **challenging** query based on the provided context. The query must be answerable only "
    "through the given context, requiring deeper analysis and understanding, rather than simple fact recall.\n"
    "Additionally, you need to ensure that the query can be answered with a response of 30 words or fewer."
    "Avoid using vague or broad language."
    "Your generated query must meet the following criteria:\n"
    "1. **Context-Dependent**: The query should only be answerable using information found in the context. "
    "It must not rely on external common knowledge.\n"
    "2. **Challenging**: The query should require a higher level of reasoning or synthesis of information from the context,"
    " not merely fact extraction."
    "   However, the query must also focus on the most **critical information** from the context, "
    "not secondary or less important details.\n"
    "3. **Independent Formulation**: The query must be phrased in a way that makes it look like an independent query "
    "(as if input into a search engine)."
    "   It must not contain phrases like 'according to the passage' or 'based on the context.'\n"
    "4. **Answerability**: The query must be answerable with a response of 30 words or fewer.\n"
    "   It must be semantically clear, easy to understand, and clearly indicate the specific answer being sought."
    "   The query should be precise and specific, asking about processes, terms, numbers, or specific information, "
    "using 'what' questions whenever possible, and 'how' when necessary."
    "### Output Instructions:\n\n"
    "   Only output the query itself, do not provide the answer.\n"
    "   Please generate the query without including any information other than the query."
    "Now here is the context.\n"
    "   Context: {context}\n"
)

GENERATE_ANSWER_PROMPT = (
    "### Task Description:\n"
    "Your task is to generate a **concise and accurate answer** based on the provided query and context. "
    "The answer must rely **exclusively** "
    "on the information found in the context and demonstrate a clear understanding of the query.\n\n"
    "Your generated answer must meet the following criteria:\n"
    "1. **Context-Dependent**: The answer should be based solely on the information from the context. "
    "It must not rely on external common knowledge, "
    "nor introduce any new information that is not present in the context.\n"
    "2. **Conciseness**: The answer should be kept within 30 words if possible, "
    "while ensuring it is precise and fully answers the query.\n"
    "3. **Logical Coherence**: The answer should be well-structured, logically consistent, "
    "and aligned with the context, following a clear line of thought.\n"
    "4. **Highlighting Key Points**: Carefully understand what the query is asking, and when generating the answer, "
    "emphasize key points using # symbols—i.e., "
    "a single word, number, or fact that must be included in the answer. No more than 2 key points should be highlighted.\n"
    "### Output Example:\n\n"
    "The tensile strength of the alloy remains at #500 MPa#, #corrosion resistance# has significantly improved.\n"
    "### Output Instructions:\n\n"
    "- Output only the answer itself; do not include the query or context.\n"
    "- Ensure the answer is concise, accurate, and directly answers the query based on the context.\n\n"
    
    "Here is the query and context.\n"
    "Query: {query}\n"
    "Context: {context}\n\n"
)

class QAGenerator(BaseQAGenerator):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator

    def test_queries_generate(self, claim_sets: list, num: int) -> list:
        test_queries = []
        for claim_set in claim_sets:
            claim_list = claim_set['Claim_set']
            if len(claim_list) == 1:
                text = claim_list[0]['text']
                raw_query = self.generator.generate(SINGLE_QUERY_PROMPT.format(context=text))
                claim_results = {
                    "raw_query": raw_query,
                    "num_of_contexts": len(claim_list),
                    "Claims": [
                        {
                            "text": claim_data["text"],
                            "Claim": claim_data["Claim"],
                            "Topic": claim_data["Topic"],
                            "chunk_idx": claim_data["chunk_idx"]
                        } for claim_data in claim_list
                    ]
                }
                test_queries.append(claim_results)
            if(len(test_queries) >= num):
                break
        return test_queries

    def processing_loop(self, claim_sets: list, optimized_single_query_prompt: str, store_file: str, batch_size: int = 10) ->None:
        batch_counter = 0 
        global json_output  
        # 检查是否已有存储文件存在，若存在则读取现有数据
        try:
            with open(store_file, "r") as json_file:
                json_output = json.load(json_file)
        except FileNotFoundError:
            json_output = []  # 文件不存在则初始化为空列表
        
        for query_idx, claim_set in enumerate(tqdm(claim_sets, desc="Processing claim sets"), start=1):  
            claim_list = claim_set['Claim_set']

            try:
                if len(claim_list) != 1:
                    claim_results = self.process_claim_set(query_idx, claim_list)
                else:
                    claim_results = self.process_single_query(query_idx, claim_list, optimized_single_query_prompt)
                
                json_output.append(claim_results)
                batch_counter += 1

                if batch_counter % batch_size == 0:
                    with open(store_file, "w") as json_file:
                        json.dump(json_output, json_file, indent=4, ensure_ascii=False)
                    print(f"Stored batch at claim_set {query_idx}")

            except Exception as e:
                print(f"Error {e} occurs when handling the response: {claim_results}")

        with open(store_file, "w") as json_file:
            json.dump(json_output, json_file, indent=4, ensure_ascii=False)

        print(f"Final storage completed at claim_set {query_idx}")

    def process_claim_set(self, query_idx: int, claim_list: list) -> dict:
        texts, topics = "", ""

        for index, claim_data in enumerate(claim_list):
            texts += f"Context {index}: {claim_data['text']}\n"
            topics += f"Topic for Context {index}: {claim_data['Topic']}\n"

        response = self.generator.generate(MULTIHOP_QUERY_PROMPT.format(contexts=texts, topic=topics))

        try:
            raw_query = re.search(r'STEP2:\s*(.*?)\nSTEP3:', response, re.DOTALL | re.IGNORECASE).group(1).strip()
            core_points = re.search(r'STEP1:\s*(.*?)\nSTEP2:', response, re.DOTALL | re.IGNORECASE).group(1).strip()
            raw_answer = re.search(r'STEP3:\s*(.*)', response, re.DOTALL | re.IGNORECASE).group(1).strip()
            refine_response = self.generator.generate(OPTIMIZE_MULTIHOP_QUERY_PROMPT.format(multihop_query=raw_query, core_points=core_points, contexts=texts, answer=raw_answer))
        except Exception as e:
            print(f"Error during parsing response of multi_hop query generation: {e}")
            print(f"The response is {response}")
            raw_query = None
            raw_answer = None   
        
        try:
            optimized_query = re.search(r'STEP2:\s*(.*?)\nSTEP3:', refine_response, re.DOTALL | re.IGNORECASE).group(1).strip()
            
            optimized_answer = re.search(r'STEP4:\s*(.*)', refine_response).group(1).strip()
        except Exception as e:
                print(f"Error during parsing optimized query for claim_set: {e}")
                print("The response that need to be addressed:\n",refine_response)
                optimized_query = None
                optimized_answer = None

        claim_results = {
            "query_idx": query_idx,
            "raw_query": raw_query,
            "optimized_query": optimized_query if optimized_query else raw_query,
            "groundtruth_answer": optimized_answer if optimized_answer else raw_answer,
            "num_of_contexts": len(claim_list),
            "Claims": [
                {
                    "text": claim_data["text"],
                    "Claim": claim_data["Claim"],
                    "Topic": claim_data["Topic"],
                    "chunk_idx": claim_data["chunk_idx"]
                } for claim_data in claim_list
            ]
        }

        return claim_results

    def process_single_query(self, query_idx: int, claim_list: list, optimized_single_query_prompt: str) -> dict:
        text = claim_list[0]['text']
    
        raw_query = self.generator.generate(SINGLE_QUERY_PROMPT.format(context=text))
        
        refine_response = self.generator.generate(optimized_single_query_prompt.format(query=raw_query, context=text))
        try:
            optimized_query = parse_steps(refine_response)[-1]['step_instruction']
            answer = self.generator.generate(GENERATE_ANSWER_PROMPT.format(query=optimized_query, context=text))

        except Exception as e:
                print(f"Error during parsing optimized query for claim_set: {e}")
                optimized_query = None
                answer = None

        claim_results = {
            "query_idx": query_idx,
            "raw_query": raw_query,
            "optimized_query": optimized_query,
            "groundtruth_answer": answer,
            "num_of_contexts": len(claim_list),
            "Claims": [
                {
                    "text": claim_data["text"],
                    "Claim": claim_data["Claim"],
                    "Topic": claim_data["Topic"],
                    "chunk_idx": claim_data["chunk_idx"]
                } for claim_data in claim_list
            ]
        }
        
        return claim_results