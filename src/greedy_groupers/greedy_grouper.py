import numpy as np
from .base_greedy_grouper import BaseGreedyGrouper

class GreedyGrouper(BaseGreedyGrouper):
    def build_claim_set(self, claim_collection: list, threshold: float = 0.65, max_group_size: int = 3) -> list:
        grouped_output = []
        # 构建相似度矩阵
        similarity_matrix = self.build_similarity_matrix(claim_collection)

        # 使用贪心算法进行分组
        grouped_indices = self.greedy_grouping(claim_collection, similarity_matrix, threshold=threshold, max_group_size=max_group_size)
        
        # 打包并存储数据
        for group_id, group in enumerate(grouped_indices, start=1):
            group_data = {
                "Claim_set_idx": group_id,
                "Claim_set": [
                    {
                        'chunk_idx': claim_collection[idx]['idx'],
                        'text': claim_collection[idx]['text'],
                        'Claim': claim_collection[idx]['Claim'],
                        'Entity': claim_collection[idx]['Entity'],
                        'Topic': claim_collection[idx]['Topic'],
                        'best_similarity': None if len(group) == 1 else max([similarity_matrix[idx][j] for j in group if j != idx]), 
                        'avg_similarity': None if len(group) == 1 else np.mean([similarity_matrix[idx][j] for j in group if j != idx])
                    } for idx in group
                ]
            }
            grouped_output.append(group_data)
        return grouped_output

    def build_similarity_matrix(self, claim_collection: list):
        n = len(claim_collection)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                similarity = self.cosine_similarity(claim_collection[i]['Claim_embedding'], claim_collection[j]['Claim_embedding'])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity 
        return similarity_matrix
    
    def greedy_grouping(self, claim_collection: list, similarity_matrix, threshold=0.7, max_group_size=3) -> list:
        n = len(claim_collection)
        grouped_data = []
        ungrouped_indices = set(range(n))  # 未分组的数据项索引

        while ungrouped_indices:
            if len(ungrouped_indices) <= max_group_size:
                # 当剩余未分组项数量少于等于max_group_size时，仍然检查它们是否满足相似度要求
                new_group = []
                for i in list(ungrouped_indices):
                    added = False
                    for j in new_group:
                        if similarity_matrix[i, j] >= threshold:
                            new_group.append(i)
                            added = True
                            break
                    if not added:
                        # 如果不满足阈值，i 作为单独的一组
                        grouped_data.append([i])
                    else:
                        ungrouped_indices.remove(i)
                if new_group:
                    grouped_data.append(new_group)
                break

            # 找到相似度最高的一对未分组数据项，且相似度大于阈值
            best_pair = None
            best_similarity = -1
            for i in ungrouped_indices:
                for j in ungrouped_indices:
                    if i != j and similarity_matrix[i, j] > best_similarity and similarity_matrix[i, j] > threshold:
                        best_similarity = similarity_matrix[i, j]
                        best_pair = (i, j)

            if best_pair is None:
                # 如果没有找到相似度大于阈值的对，则将剩下的每个项作为单独的组
                for idx in ungrouped_indices:
                    grouped_data.append([idx])
                break

            # 创建新组，包含最佳对
            new_group = [best_pair[0], best_pair[1]]
            ungrouped_indices.remove(best_pair[0])
            ungrouped_indices.remove(best_pair[1])

            # 找到与当前组最相似的第三个数据项（如果还有空间），且相似度大于阈值
            if len(new_group) < max_group_size:
                best_third_item = None
                best_third_similarity = -1
                for k in ungrouped_indices:
                    avg_similarity = np.mean([similarity_matrix[k, idx] for idx in new_group])
                    if avg_similarity > best_third_similarity and avg_similarity > threshold:
                        best_third_similarity = avg_similarity
                        best_third_item = k

                if best_third_item is not None:
                    new_group.append(best_third_item)
                    ungrouped_indices.remove(best_third_item)

            grouped_data.append(new_group)

        return grouped_data
    
    def cosine_similarity(self, emb_a, emb_b):
        return np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))