from abc import ABC, abstractmethod

class BaseNodesGenerator(ABC):
    @abstractmethod
    def generate_nodes(self, dataset_path, chunksize, chunk_overlap):
        pass