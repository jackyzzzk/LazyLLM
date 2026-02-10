"""
Reranker From Embedding Converter Operator

This operator converts embedding training data to reranker format.
该算子将 Embedding 训练数据转换为 Reranker 格式。
"""
import json
import random
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


class RerankerFromEmbeddingConverter(reranker):
    """
    Convert embedding training data to reranker format.
    将 Embedding 训练数据转换为 Reranker 格式。

    The main difference between embedding and reranker training data:
    - Embedding: {"query": str, "pos": [str], "neg": [str], "prompt": str}
    - Reranker: {"query": str, "pos": [str], "neg": [str]} (no prompt/instruction)

    This operator:
    1. Removes the prompt/instruction field
    2. Adjusts negative count to match train_group_size
    3. Pads negatives from duplicates if needed

    Args:
        remove_instruction: Whether to remove instruction/prompt fields (default: True)
        adjust_neg_count: Target number of negatives per sample (default: 7)
        output_file: Path to save converted data (optional)
        seed: Random seed for reproducibility (default: 42)
    """

    def __init__(
            self,
            remove_instruction: bool = True,
            adjust_neg_count: int = 7,
            output_file: Optional[str] = None,
            seed: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.remove_instruction = remove_instruction
        self.adjust_neg_count = adjust_neg_count
        self.output_file = output_file
        self.seed = seed
        LOG.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RerankerFromEmbeddingConverter 算子用于将 Embedding 训练数据转换为 Reranker 格式。\n\n"
                "核心功能：\n"
                "- 移除 prompt/instruction 字段\n"
                "- 调整负样本数量以匹配 train_group_size\n"
                "- 保持 query-pos-neg 格式\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- input_pos_key: 正样本字段名（默认：'pos'）\n"
                "- input_neg_key: 负样本字段名（默认：'neg'）\n\n"
                "输出：Reranker 格式的训练数据列表"
            )
        else:
            return (
                "RerankerFromEmbeddingConverter converts embedding training data to reranker format.\n\n"
                "Features:\n"
                "- Removes prompt/instruction fields\n"
                "- Adjusts negative count to match train_group_size\n"
                "- Maintains query-pos-neg format"
            )

    def _save_to_file(self, result: dict):
        """Save a single result to output file if specified."""
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def forward(
            self,
            data,
            input_query_key: str = "query",
            input_pos_key: str = "pos",
            input_neg_key: str = "neg",
    ):
        """
        Convert a single embedding sample to reranker format.

        Args:
            data: Dict containing query, pos, neg (and optionally prompt) fields
            input_query_key: Key for query field
            input_pos_key: Key for positive samples field
            input_neg_key: Key for negative samples field

        Returns:
            Dict in reranker format (without prompt/instruction)
        """
        assert isinstance(data, dict), "Input data must be a dict"

        query = data.get(input_query_key, "")
        pos = data.get(input_pos_key, [])
        neg = data.get(input_neg_key, [])

        if not query:
            LOG.warning("Empty query in data, skipping")
            return []  # 返回空列表表示删除该数据

        # Ensure pos is a list
        if not isinstance(pos, list):
            pos = [pos] if pos else []

        # Ensure neg is a list
        if not isinstance(neg, list):
            neg = [neg] if neg else []

        # Adjust negative count
        if len(neg) > self.adjust_neg_count:
            # Truncate to target count
            neg = neg[:self.adjust_neg_count]
        elif len(neg) < self.adjust_neg_count and neg:
            # Pad with duplicates if needed (when we have some negatives)
            random.seed(self.seed)
            while len(neg) < self.adjust_neg_count:
                neg.append(random.choice(neg))

        # Build reranker format (no prompt/instruction)
        reranker_item = {
            "query": query,
            "pos": pos,
            "neg": neg,
        }

        # Save to file if specified
        self._save_to_file(reranker_item)

        return reranker_item
