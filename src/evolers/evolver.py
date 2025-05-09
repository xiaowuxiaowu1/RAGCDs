from .base_evolver import BaseEvolver
from src.generators.base_generator import BaseGenerator

INITIAL_EVOLVE_METHOD = (
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
    
    "Step 1: Carefully read the '#Query#' and '#context#' below, and list all possible methods "
    "to make the query more ambiguous and harder to directly match with the original context, "
    "thereby increasing the chances of matching with other contexts.\n"
    "Additionally, make the query more challenging for powerful models (like gpt-4o) to answer."
    "However, ensure that the modified query can still be fully answered using the context, "
    "and avoid creating unanswerable queries.\n"
    "Do not suggest changes to the language of the instruction!\n"
    
    "Step 2: Based on the #Methods List# generated in Step 1, create a detailed plan "
    "to make the #Query# more complex while ensuring it can still be answered using the context."
    "The query should be precise and specific, asking about processes, terms, numbers, or specific information, "
    "preferably using 'what' questions, though 'how' may also be used when necessary."
    "The plan should incorporate several methods from the #Methods List#.\n"
    
    "Step 3: Execute the plan step by step and provide the #Rewritten Query#."
    "#Rewritten Query# must be harder to directly match with the original context.\n"
    "At the same time, make it more challenging for powerful models (like gpt-4o) to answer the query."
    
    "Step 4: Ensure the #Rewritten Query# can still be answered using the context.\n"
    "Ensure the #Rewritten Query# explicitly targets the required answer, meaning that the answer it seeks is clear."
    
    "Step 5: Carefully review the #Rewritten Query# and identify any unreasonable parts based on the earlier requirements.\n"
    "Only provide the #Finally Rewritten Query#, and do not provide any explanations."

    "\n#Query#:\n {query}\n"
    "#context#:\n {context}\n"

    "Remember, your task is to generate a more complex version of the query, not to answer the #Query#.\n"
    "The #Finally Rewritten Query# should be harder to directly match with the original context, "
    "but it must still be answerable using the context.\n"

    "**Output Instructions**"
    "Please generate the optimized query strictly following the format below, and do not add anything else:\n"

    "```Optimized Instruction\n"
    "Step 1:\n#Methods List#\n"
    "Step 2:\n#Plan#\n"
    "Step 3:\n#Rewritten Query#\n"
    "Step 4:\n#Verified Query#\n"
    "Step 5:\n#Finally Rewritten Query#\n"
    "```"
)

class Evolver(BaseEvolver):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator

    def evolve(self, query: str, context: str, evolving_method: str = None) -> tuple:
        evol_method = evolving_method if evolving_method else INITIAL_EVOLVE_METHOD
        evol_method = evol_method.format(query=query, context=context)
        return self.generator.generate(evol_method), evol_method
    