from .base_optimizer import BaseOptimizer
from src.generators import BaseGenerator
from src.utils import parse_steps

OPTIMIZE_PROMPT = (
    "Feedback: {feedback}"
    "You are an Instruction Method Optimizer."
    "Based on the feedback from the evolution failure case, optimize the method below to create a more effective instruction rewriting process,"
    "The new process must be applicable to different cases, so the process should not include methods specific to any particular case."
    "while ensuring that performance on other cases is not negatively impacted."
    "Ensure that the complexity of the optimized method is not lower than the previous method."
    "If the feedback is '### PASSED', then come up with a better method than the current one"
    "to create a more effective instruction rewriting process that meets the requirements."
    "Remember, the new method should not be very similar to the current method, be creative with new steps for the new method."

    "Current Method:"
    "{current_method}"

    "**Output Instructions**"
    "Add more steps to achieve the most refined method if needed; however,"
    "REMEMBER that the final step in your output has to be '#Finally Rewritten Query#', no matter how many steps are added."
    "Please strictly generate the optimized method using ONLY the format below, do not add anything else:"
    "Re-emphasis: The new process must be applicable to different cases, so the process should not include methods specific to any particular case."
    "```Optimized Method"
    "Step 1:"
    "#Methods List#"
    "Describe how to generate a list of methods to make the query more complex"
    "and harder to directly match with the context,"
    "thereby increasing the chances of matching with other contexts."
    "At the same time, make sure the rewritten query can still be fully answered"
    "using the context, and avoid creating an unanswerable query."
    "Incorporate the feedback."

    "Step 2:"
    "#Plan#"
    "Explain how to create a comprehensive plan based on the Methods List."

    "[This is a Note, do not output it] Add more steps here if needed to achieve the best method. The steps should align with the instruction domain/topic,"
    "and should not involve any tools or visualization, they should be text-only methods."
    "The last step should always be #Finally Rewritten Query#."
    "Replace N with the actual number of steps you have used."

    "Step N-2:"
    "#Rewritten Query#"
    "Do not generate a new Query here, but please provide a detailed process of executing the plan to rewrite the query."
    "You are generating a guide to write a better query, NOT THE QUERY ITSELF."

    "Step N-1:"
    "#Verified Query#"
    "Do not generate a new Query here, but please provide a detailed process to verify that the rewritten query can still be fully answered using the context."
    "You are generating a guide to verify the rewritten query, NOT THE QUERY ITSELF."

    "Step N:"
    "#Finally Rewritten Query#"
    "Do not generate a new Query here, but please provide the process to write the final rewritten query."
    "You are generating a guide to write a better query, NOT THE QUERY ITSELF."
    "```"
)

EVOLVE_METHOD = (
    "You are a query rewriter tasked with rewriting the given #Query# into a more complex version.\n"
    "The goal is to make the query more ambiguous and harder to directly match with the original context, "
    "thus increasing the chances of matching with other contexts."
    "At the same time, it should be more challenging for powerful models (like gpt-4o) to answer the query."
    "However, the modified query must still be fully answerable using the context, "
    "and it must be semantically clear, easy to understand, and explicitly target the required answer."
    "The query should be precise and specific, asking about processes, terms, numbers, or specific information, "
    "and prefer using 'what' questions, though 'how' may also be used when necessary."
    "You must avoid creating unanswerable queries.\n"
    "Please follow the steps below to rewrite the given '#Query#' into a more complex version.\n"

    "{steps}\n"
    "#Query#: \n{query}\n"
    "#context#: \n{context}\n"

    "Remember, your task is to generate a more complex version of the query, not to answer the #Query#.\n"
    "The #Finally Rewritten Query# should be harder to match directly with the original context, "
    "but it must still be answerable using the context.\n"

    "**Output Instructions**"
    "Please generate the optimized query strictly using ONLY the given below format, do not add anything else:\n"

    "```Optimized Instruction"
    "{format_steps}"
    "```"
)

class Optimizer(BaseOptimizer):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator

    def optimize(self, feedback: str, current_method: str) -> str:
        optimize_prompt = OPTIMIZE_PROMPT.format(feedback=feedback, current_method=current_method)
        optimized_method = self.generator.generate(optimize_prompt)
        parsed_steps = parse_steps(optimized_method)
        optimized_instruction = self.build_my_method(parsed_steps)
        return optimized_instruction

    def build_my_method(self, parsed_steps: list) -> str:
        step_details = ""
        format_steps = ""
        
        for i, step in enumerate(parsed_steps, start=1):
            step_name = step['step_name']
            step_instruction = step['step_instruction']
                
            step_details += f"Step {i}: {step_instruction}\n\n"
            format_steps += f"Step {i}:\n#{step_name}#\n\n"
        
        return EVOLVE_METHOD.format(steps=step_details.strip(), format_steps=format_steps, query='{query}', context='{context}')


        

