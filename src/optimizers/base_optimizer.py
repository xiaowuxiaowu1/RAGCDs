from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self, feedback, current_method):
        pass
