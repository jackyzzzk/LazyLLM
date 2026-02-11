"""
Embedding Data Formatter Operators

This module provides operators for formatting embedding training data into standard formats.
"""
import json
import random
from pathlib import Path
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# Get or create embedding group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


class EmbeddingFormatFlagEmbedding(embedding):
    """
    Format data to FlagEmbedding/BGE training format.
    Output: {"query": str, "pos": [str], "neg": [str], "prompt": str}
    """

    def __init__(self, instruction: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.instruction = instruction

    def forward(self, data: dict) -> dict:
        """Format single item to FlagEmbedding format."""
        query = data.get('query', '')
        pos = data.get('pos', [])
        neg = data.get('neg', [])

        if not query or not pos:
            return []

        # Ensure pos and neg are lists
        if not isinstance(pos, list):
            pos = [pos]
        if not isinstance(neg, list):
            neg = [neg] if neg else []

        result = {
            "query": query,
            "pos": pos,
            "neg": neg,
        }
        if self.instruction:
            result["prompt"] = self.instruction

        return result


class EmbeddingFormatSentenceTransformers(embedding):
    """
    Format data to Sentence-Transformers training format.
    Output: List[{"anchor": str, "positive": str, "negative": str}]
    """

    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict) -> List[dict]:
        """Format single item to Sentence-Transformers format (returns list)."""
        query = data.get('query', '')
        pos = data.get('pos', [])
        neg = data.get('neg', [])

        if not query or not pos:
            return []

        # Ensure pos and neg are lists
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg] if neg else []

        # Create anchor-positive-negative triplets
        results = []
        for p in pos_list:
            for n in neg_list:
                results.append({
                    "anchor": query,
                    "positive": p,
                    "negative": n,
                })

        return results


class EmbeddingFormatTriplet(embedding):
    """
    Format data to simple triplet format.
    Output: List[{"query": str, "positive": str, "negative": str}]
    """

    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict) -> List[dict]:
        """Format single item to triplet format (returns list)."""
        query = data.get('query', '')
        pos = data.get('pos', [])
        neg = data.get('neg', [])

        if not query or not pos:
            return []

        # Ensure pos and neg are lists
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg] if neg else []

        # Create query-positive-negative triplets
        results = []
        for p in pos_list:
            for n in neg_list:
                results.append({
                    "query": query,
                    "positive": p,
                    "negative": n,
                })

        return results


class EmbeddingDataFormatter(embedding):
    """
    Format embedding training data into standard formats using pipeline operators.
    """

    def __init__(
        self,
        output_format: str = "flagembedding",
        instruction: Optional[str] = None,
        output_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.output_format = output_format
        self.instruction = instruction
        self.output_file = output_file
        LOG.info(f"Initializing {self.__class__.__name__} with format: {output_format}")


    def forward_batch_input(
        self,
        inputs: List[dict],
        input_query_key: str = "query",
        input_pos_key: str = "pos",
        input_neg_key: str = "neg",
        **kwargs
    ) -> List[dict]:
        """Format the training data using pipeline operators."""
        from lazyllm import pipeline

        assert isinstance(inputs, list), "inputs must be a list of dict"

        LOG.info(f"Formatting {len(inputs)} samples to {self.output_format} format...")

        # Normalize input data
        normalized_inputs = []
        for item in inputs:
            normalized_item = {
                'query': item.get(input_query_key, ''),
                'pos': item.get(input_pos_key, []),
                'neg': item.get(input_neg_key, []),
            }
            # Skip items with missing query or pos
            if normalized_item['query'] and normalized_item['pos']:
                normalized_inputs.append(normalized_item)
            else:
                LOG.warning(f"Skipping item with missing query or pos: {item}")

        # Use appropriate formatter based on output format
        with pipeline() as ppl:
            if self.output_format == "flagembedding":
                ppl.format = EmbeddingFormatFlagEmbedding(instruction=self.instruction)
            elif self.output_format == "sentence_transformers":
                ppl.format = EmbeddingFormatSentenceTransformers()
            elif self.output_format == "triplet":
                ppl.format = EmbeddingFormatTriplet()
            else:
                raise ValueError(f"Unknown output format: {self.output_format}")

        results = ppl(normalized_inputs)

        LOG.info(f"Formatted {len(results)} training samples.")

        # Save to file if specified
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            LOG.info(f"Saved formatted data to {output_path}")

        return results


class EmbeddingTrainTestSplitter(embedding):
    """
    Split embedding training data into train/test sets.
    """

    def __init__(
        self,
        test_size: float = 0.1,
        seed: int = 42,
        stratify_key: Optional[str] = None,
        train_output_file: Optional[str] = None,
        test_output_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.test_size = test_size
        self.seed = seed
        self.stratify_key = stratify_key
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        LOG.info(f"Initializing {self.__class__.__name__} with test_size: {test_size}")

    def forward_batch_input(
        self,
        inputs: List[dict],
        **kwargs
    ) -> List[dict]:
        """Split the data into train and test sets."""
        assert isinstance(inputs, list), "inputs must be a list of dict"

        LOG.info(f"Splitting {len(inputs)} samples with test_size={self.test_size}")

        # Shuffle and split
        random.seed(self.seed)
        shuffled = inputs.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - self.test_size))
        train_data = shuffled[:split_idx]
        test_data = shuffled[split_idx:]

        # Add split labels
        for item in train_data:
            item['split'] = 'train'
        for item in test_data:
            item['split'] = 'test'

        LOG.info(f"Split completed: {len(train_data)} train, {len(test_data)} test")

        # Save to files if specified
        if self.train_output_file:
            output_path = Path(self.train_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in train_data:
                    item_copy = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f"Saved train data to {output_path}")

        if self.test_output_file:
            output_path = Path(self.test_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in test_data:
                    item_copy = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f"Saved test data to {output_path}")

        return train_data + test_data
