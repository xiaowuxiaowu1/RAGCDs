from abc import ABC, abstractmethod

class BaseEvolver(ABC):
    @abstractmethod
    def evolve(self, query, context, evolving_method):
        pass