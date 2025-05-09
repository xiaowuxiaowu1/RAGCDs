from .base_judge import BaseJudge
from typing import Any, List, Tuple, Dict
from src.generators import BaseGenerator
from tqdm import tqdm
JUDGEMENT_PROMPT = (
    "As an expert in materials science and chemistry, "
    "you are tasked with evaluating whether the given text fragment is suitable for use in examinations for PhD students. "
    "Please assess the text based on the following standards: "
    
    "1. **Technical Terms and Concepts**: The text must include relevant technical terms and concepts from chemistry or materials science. "
    "These concepts should be applied or explained in context. If the text simply stacks concepts without explanation, judge it as 'No'. "
    
    "2. **Applicability of Knowledge**: Determine whether the text contains validated knowledge that can be applied to research or experimental design. "
    
    "3. **Independence of Understanding**: Assess whether the text can be independently understood as a coherent sentence. "
    
    "4. **Clarity and Completeness**: If the text contains excessive Markdown symbols or lacks a comprehensive expression of ideas, mark it as 'No'. "
    
    "For your decision, please provide a concise reason (not exceeding 15 words) and a clear judgment of 'Yes' or 'No'. "
    
    "##Output Example##:\n"
    "Reason: ... "
    "Judgment: Yes / No\n"
    "##Text Fragment##:"
    "{text}\n"
    "Now, it’s your turn."
)

class Judge(BaseJudge):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator
    
    def judge(self, nodes: list) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        reference_chunks = []
        failed_chunks = []

        for node in tqdm(nodes, desc="Judging nodes"):
            judgement_prompt = JUDGEMENT_PROMPT.format(text=node.get_content())
            eval_result = self.generator.generate(judgement_prompt)
            # 存入结果
            result_entry = {
                "text": node.get_content(),
                "metadata": {},
                "reason": eval_result  # 将eval_result存入reason字段
            }
            
            for key, value in node.metadata.items():
                result_entry["metadata"][key] = value
            
            if "Yes" in eval_result:
                reference_chunks.append(result_entry)  # 存入reference_chunks
            else:
                failed_chunks.append(result_entry)  # 存入failed_chunks
                
        for idx, chunk in enumerate(reference_chunks, start=1):
            chunk['idx'] = idx

        return reference_chunks, failed_chunks
