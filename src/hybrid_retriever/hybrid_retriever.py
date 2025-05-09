# import QueryBundle
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever
)

from typing import List, Dict

class HybridRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        top_k: int = 5,
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # Retrieve results from both retrievers
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        # Extract the chunk_idx (or doc_id) for both results
        combined_dict = {n.node.metadata["chunk_idx"]: n for n in vector_nodes}
        combined_dict.update({n.node.metadata["chunk_idx"]: n for n in bm25_nodes})

        # Get scores from both retrievers
        vector_scores = {n.node.metadata["chunk_idx"]: n.score for n in vector_nodes}
        bm25_scores = {n.node.metadata["chunk_idx"]: n.score for n in bm25_nodes}

        # Normalize both sets of scores
        normalized_vector_scores = self.min_max_norm(vector_scores)
        normalized_bm25_scores = self.min_max_norm(bm25_scores)

        # Combine scores by averaging if present in both, otherwise use the available score
        final_nodes = []
        for rid in combined_dict.keys():
            if rid in normalized_vector_scores and rid in normalized_bm25_scores:
                combined_score = normalized_vector_scores[rid] + normalized_bm25_scores[rid]
            elif rid in normalized_vector_scores:
                combined_score = normalized_vector_scores[rid]
            else:
                combined_score = normalized_bm25_scores[rid]

            # Update the score of the node and add it to the final list
            node = combined_dict[rid]
            node.score = combined_score
            final_nodes.append(node)

        final_nodes = sorted(final_nodes, key=lambda x: x.score, reverse=True)[:self._top_k]
        
        return final_nodes

    def min_max_norm(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalization of the scores."""
        if not scores:
            return {}
        
        min_score = min(scores.values())
        max_score = max(scores.values())
        denominator = max(max_score - min_score, 1e-9)

        return {doc_id: (score - min_score) / denominator for doc_id, score in scores.items()}