import json
import random
import os
from pathlib import Path
from typing import Optional, List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 获取或创建 reranker 分组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


class RerankerValidateEmbeddingData(reranker):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(
        self,
        data: dict,
        input_query_key: str = "query",
        input_pos_key: str = "pos",
        input_neg_key: str = "neg",
        **kwargs
    ) -> dict:
        query = data.get(input_query_key, "")
        pos = data.get(input_pos_key, [])

        if not query:
            return {**data, '_is_valid': False, '_error': 'Empty query'}

        # Ensure pos is a list
        if not isinstance(pos, list):
            pos = [pos] if pos else []

        if not pos:
            return {**data, '_is_valid': False, '_error': 'No positive samples'}

        # Ensure neg is a list
        neg = data.get(input_neg_key, [])
        if not isinstance(neg, list):
            neg = [neg] if neg else []

        return {
            **data,
            '_is_valid': True,
            '_query': query,
            '_pos': pos,
            '_neg': neg
        }


class RerankerAdjustNegatives(reranker):
    def __init__(self, adjust_neg_count: int = 7, seed: int = 42, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.adjust_neg_count = adjust_neg_count
        self.seed = seed

    def forward(self, data: dict, **kwargs) -> dict:
        if not data.get('_is_valid'):
            return data

        neg = data.get('_neg', [])

        if len(neg) > self.adjust_neg_count:
            # Truncate to target count
            neg = neg[:self.adjust_neg_count]
        elif len(neg) < self.adjust_neg_count and neg:
            # Pad with duplicates if needed (when we have some negatives)
            local_random = random.Random(f"{self.seed}_{data['_query']}")
            while len(neg) < self.adjust_neg_count:
                neg.append(local_random.choice(neg))

        return {**data, '_neg': neg}


class RerankerBuildFormat(reranker):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict, **kwargs) -> dict:
        if not data.get('_is_valid'):
            return {}

        # Build reranker format (no prompt/instruction)
        reranker_item = {
            "query": data['_query'],
            "pos": data['_pos'],
            "neg": data['_neg'],
        }

        return reranker_item


class RerankerSaveConverted(reranker):
    def __init__(self, output_file: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_file = output_file

    def forward(self, data: dict, **kwargs) -> dict:
        if not self.output_file or not data:
            return data

        try:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            LOG.warning(f"Failed to save to file: {e}")

        return data


class RerankerFromEmbeddingConverter(reranker):
    def __init__(
            self,
            remove_instruction: bool = True,
            adjust_neg_count: int = 7,
            output_file: Optional[str] = None,
            seed: int = 42,
            **kwargs
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.remove_instruction = remove_instruction
        self.adjust_neg_count = adjust_neg_count
        self.output_file = output_file
        self.seed = seed
        LOG.info(f"Initializing {self.__class__.__name__}...")

    def forward_batch_input(
            self,
            data: List[dict],
            input_query_key: str = "query",
            input_pos_key: str = "pos",
            input_neg_key: str = "neg",
    ) -> List[dict]:
        from lazyllm import pipeline

        assert isinstance(data, list), "Input data must be a list of dict"

        LOG.info(f"Converting {len(data)} embedding samples to reranker format...")

        # Build parallel processing pipeline
        with pipeline() as ppl:
            # Stage 1: Validate data (CPU-bound)
            ppl.validate = RerankerValidateEmbeddingData(
                input_query_key=input_query_key,
                input_pos_key=input_pos_key,
                input_neg_key=input_neg_key
            )
            
            # Stage 2: Adjust negative count (CPU-bound)
            ppl.adjust = RerankerAdjustNegatives(
                adjust_neg_count=self.adjust_neg_count,
                seed=self.seed
            )
            
            # Stage 3: Build reranker format (CPU-bound)
            ppl.build = RerankerBuildFormat()
            
            # Stage 4: Save to file (I/O-bound)
            if self.output_file:
                ppl.save = RerankerSaveConverted(output_file=self.output_file)

        # Execute pipeline
        results = ppl(data)

        # Filter out invalid results
        valid_results = [r for r in results if r]

        LOG.info(f"Converted {len(valid_results)} valid samples to reranker format")
        return valid_results
