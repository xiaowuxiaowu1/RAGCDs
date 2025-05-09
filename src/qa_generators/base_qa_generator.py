from abc import ABC, abstractmethod

class BaseQAGenerator(ABC):
    @abstractmethod
    def process_claim_set(self, query_idx, claim_set):
        pass

    @abstractmethod
    def process_single_query(self, query_idx, claim_list):
        pass

    def processing_loop(claim_sets, store_file, batch_size):
        pass

    