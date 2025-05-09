import re
from src.utils import create_AzureOpenAIEmbedding_3_large
from .base_extractor import BaseExtractor
from src.generators import BaseGenerator
from tqdm import tqdm

EXTRACT_PROMPT = (
    "Please analyze the given text and extract the following information:"
    "1. Topic: Extract the central argument concept of the text, ensuring it is a concise phrase representing the main idea."
    "2. Entity: Identify and extract the most critical object of discussion in the text, ensuring it is a single, refined term."
    "3. Claim: Extract a highly relevant claim related to the topic and entity, requiring:"
    "   - The claim should be built around the extracted topic and entity, clearly expressing a belief or fact."
    "   - Use the original keywords from the text to ensure fidelity, avoiding any significant alterations or semantic discrepancies."
    "   - The content should condense the knowledge points from the text as closely as possible to the original."
    "Please follow this format for the output:"
    "Claim: [extracted claim]"
    "Entity: [most critical entity]"
    "Topic: [topic]"
    "## Given text:"
    "{text}"
    "Now, it’s your turn."
)

class Extractor(BaseExtractor):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator
        self.embed_model = create_AzureOpenAIEmbedding_3_large()

    def extract(self, reference_chunks: list) -> list:
        claim_collection = []
        for chunk in tqdm(reference_chunks, desc="Extracting cliams from reference_chunks"):
            extract_prompt = EXTRACT_PROMPT.format(text=chunk["text"])
            eval_result = self.generator.generate(extract_prompt)
            # 正则表达式提取
            try:
                claim_match = re.search(r'Claim:\s*(.*)', eval_result)
                entity_match = re.search(r'Entity:\s*(.*)', eval_result)
                topic_match = re.search(r'Topic:\s*(.*)', eval_result)

                result_entry = {
                    "idx": chunk["idx"],
                    "text": chunk["text"],
                    "Claim": claim_match.group(1).strip(),
                    "Entity": entity_match.group(1).strip(),
                    "Topic": topic_match.group(1).strip(),
                    "Claim_embedding": self.embed_model.get_text_embedding(claim_match.group(1).strip())
                }

                claim_collection.append(result_entry)

            except Exception as e:
                print(f"Error: {e} occurs，when extract {eval_result}")
        
        return claim_collection
