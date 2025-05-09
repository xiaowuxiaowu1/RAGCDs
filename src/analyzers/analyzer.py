from .base_analyzer import BaseAnalyzer
from src.generators import BaseGenerator

ANALYZE_PROMPT = (
    "You are an expert in analyzing the evolution of queries based on the context.\n"

    "You will examine the process in which a query evolves from its initial state to a more complex version."
    "The goal is to make the query harder to directly match with the original context, thereby increasing the chances of matching with other contexts,"
    
    "while also making it more challenging for powerful models (such as gpt-4o) to answer the query."
    "However, the query must still be fully answerable using the context,"
    "and it must be semantically clear, easy to understand, and explicitly target the required answer."
    "The query should be precise and specific, asking about processes, terms, numbers, or specific information, preferably using 'what' questions, though 'how' can also be used if necessary."
    "You must avoid creating unanswerable queries.\n"

    "The following list shows cases where the query evolves into a more complex version."
    "For each case, stage 0 represents the query in its initial state, and stage 1 requires an increase in complexity based on the previous stage."
    "Please identify any cases that failed to evolve correctly, and provide the reason for the failure.\n"
    "Please output strictly following the format below and do not add anything beyond what is required by the format:"

    "***FORMAT INSTRUCTION***"
    "Choose one of the two options:"
    "Option 1 - If all cases have evolved correctly, please strictly output:"
    "### PASSED"

    "Option 2 - If you identify any cases that failed to evolve correctly, please strictly output:"
    "### FAILED - Reason: [reason_of_fail]"
    "***END OF FORMAT INSTRUCTION***"
    "Evolution Trajectory:"
    "{evol_trajectory}"
    "The context on which the query is based:"
    "{context}"
)

class Analyzer(BaseAnalyzer):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator

    def analyze(self, query, context, evolved_instructions) -> str:
        trajectory_str = f"""
            Stage 0: {query}
            Stage 1: {evolved_instructions}
        """
        analyze_prompt = ANALYZE_PROMPT.format(evol_trajectory=trajectory_str, context=context)
        return self.generator.generate(analyze_prompt)
