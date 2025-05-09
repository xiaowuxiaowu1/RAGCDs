from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def calculate_score_without_context(self, queries, llm_to_pre, model_map):
        pass

    @abstractmethod
    def calculate_score_with_context(self, queries, retriever, top_k, llm_to_pre, model_map):
        pass

    @abstractmethod
    def calculate_score_with_true_context(self, queries, llm_to_pre, model_map):
        pass