import os
from llama_index.core import Document
from .base_nodes_generator import BaseNodesGenerator
from llama_index.core.node_parser import SentenceSplitter

class NodesGenerator(BaseNodesGenerator):
    def generate_nodes(self, dataset_path: str = './data/dataset', chunk_size: int = 200, chunk_overlap: int = 20) -> list:
        parser = SentenceSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        docs = self.process_all_files(dataset_path)
        nodes = parser.get_nodes_from_documents(docs)
        return nodes
    
    def process_all_files(self, directory_path):
        all_documents = []

        for file_name in os.listdir(directory_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(directory_path, file_name)
                documents = self.process_file(file_path)
                all_documents.append(documents) 
    
        return all_documents

    def process_file(self, file_path):
        with open(file_path,'r',encoding='utf=8') as file:
            content = file.read()
            file_name = os.path.basename(file_path).replace('.txt', '')
            doc = Document(
                text=content,
                metadata={"paper_title": file_name},
                metadata_template="{{{key}:{value}}}", 
                text_template="Metadata: {metadata_str}-------->Content: {content}",
            )
        
        return doc
