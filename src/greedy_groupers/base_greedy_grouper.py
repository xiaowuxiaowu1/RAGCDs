from abc import ABC, abstractmethod

class BaseGreedyGrouper(ABC):
    @abstractmethod
    def build_similarity_matrix(self, claim_collection):
        pass
    
    @abstractmethod
    def greedy_grouping(self, claim_collection, similarity_matrix, threshold, max_group_size):
        pass

    @abstractmethod
    def build_claim_set(self):
        pass
