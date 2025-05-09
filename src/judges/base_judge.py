from abc import ABC, abstractmethod

class BaseJudge(ABC):
    @abstractmethod
    def judge(self, nodes):
        pass
        