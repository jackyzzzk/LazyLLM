"""
Reranker Data Formatter Operator

This operator formats reranker training data into standard training formats.
该算子将 Reranker 训练数据格式化为标准的训练格式。
"""
import json
import random
from pathlib import Path
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 获取或创建 reranker 分组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


class RerankerDataFormatter(reranker):
    """
    Format reranker training data into standard formats.
    将 Reranker 训练数据格式化为标准格式。

    Supported output formats:
    - flagreranker: Format for FlagEmbedding Reranker training
      {"query": str, "pos": [str], "neg": [str]}
    - cross_encoder: Format for Cross-Encoder/Sentence-Transformers
      {"query": str, "document": str, "label": int}
    - pairwise: Pairwise format for learning to rank
      {"query": str, "doc_pos": str, "doc_neg": str}

    Args:
        output_format: Target output format (default: "flagreranker")
        output_file: Path to save formatted data (optional)
        train_group_size: Number of samples per training group (default: 8, i.e., 1 pos + 7 neg)
    """

    def __init__(
            self,
            output_format: str = "flagreranker",
            output_file: Optional[str] = None,
            train_group_size: int = 8,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.output_format = output_format
        self.output_file = output_file
        self.train_group_size = train_group_size
        LOG.info(f"Initializing {self.__class__.__name__} with format: {output_format}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RerankerDataFormatter 算子用于将训练数据格式化为 Reranker 标准格式。\n\n"
                "支持的输出格式：\n"
                "- flagreranker: FlagEmbedding Reranker 训练格式\n"
                "- cross_encoder: Cross-Encoder 训练格式\n"
                "- pairwise: 成对学习排序格式\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- input_pos_key: 正样本字段名（默认：'pos'）\n"
                "- input_neg_key: 负样本字段名（默认：'neg'）\n\n"
                "输出：格式化后的训练数据列表"
            )
        else:
            return (
                "RerankerDataFormatter formats training data into reranker standard formats.\n\n"
                "Supported formats:\n"
                "- flagreranker: FlagEmbedding Reranker training format\n"
                "- cross_encoder: Cross-Encoder training format\n"
                "- pairwise: Pairwise learning to rank format"
            )

    def _format_flagreranker(self, query: str, pos: List[str], neg: List[str]) -> dict:
        """Format to FlagReranker training format."""
        # Ensure neg has exactly train_group_size - 1 samples
        num_neg_needed = self.train_group_size - 1
        if len(neg) < num_neg_needed:
            # Pad with duplicates if needed
            neg = (neg * (num_neg_needed // len(neg) + 1))[:num_neg_needed] if neg else []
        else:
            neg = neg[:num_neg_needed]

        return {
            "query": query,
            "pos": pos if isinstance(pos, list) else [pos],
            "neg": neg,
        }

    def _format_cross_encoder(self, query: str, pos: List[str], neg: List[str]) -> List[dict]:
        """Format to Cross-Encoder training format (multiple rows)."""
        results = []
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg]

        # Positive samples with label 1
        for p in pos_list:
            results.append({"query": query, "document": p, "label": 1})

        # Negative samples with label 0
        for n in neg_list:
            results.append({"query": query, "document": n, "label": 0})

        return results

    def _format_pairwise(self, query: str, pos: List[str], neg: List[str]) -> List[dict]:
        """Format to pairwise learning to rank format."""
        results = []
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg]

        # Create pairwise comparisons
        for p in pos_list:
            for n in neg_list:
                results.append({"query": query, "doc_pos": p, "doc_neg": n})

        return results

    def _save_to_file(self, results: List[dict]):
        """Save results to output file if specified."""
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'a', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def forward(self, data, input_query_key: str = "query", input_pos_key: str = "pos", input_neg_key: str = "neg"):
        """
        Format a single training sample.

        Args:
            data: Dict containing query, pos, and neg fields
            input_query_key: Key for query field
            input_pos_key: Key for positive samples field
            input_neg_key: Key for negative samples field

        Returns:
            List of dict in the specified output format
            (may return multiple items for cross_encoder and pairwise formats)
        """
        assert isinstance(data, dict), "Input data must be a dict"

        query = data.get(input_query_key, "")
        pos = data.get(input_pos_key, [])
        neg = data.get(input_neg_key, [])

        if not query or not pos:
            LOG.warning(f"Skipping row with missing query or pos")
            return []

        # Ensure pos and neg are lists
        if not isinstance(pos, list):
            pos = [pos]
        if not isinstance(neg, list):
            neg = [neg] if neg else []

        # Format based on output format
        if self.output_format == "flagreranker":
            formatted = self._format_flagreranker(query, pos, neg)
            results = [formatted]
        elif self.output_format == "cross_encoder":
            results = self._format_cross_encoder(query, pos, neg)
        elif self.output_format == "pairwise":
            results = self._format_pairwise(query, pos, neg)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")

        # Save to file if specified
        self._save_to_file(results)

        return results


class RerankerTrainTestSplitter(reranker):
    """
    Split reranker training data into train/test sets.
    将 Reranker 训练数据分割为训练集和测试集。

    This operator requires batch context to perform global shuffle and split,
    so it uses forward_batch_input.

    Args:
        test_size: Proportion of data for test set (default: 0.1)
        seed: Random seed for reproducibility
        train_output_file: Path to save train data (optional)
        test_output_file: Path to save test data (optional)
    """

    def __init__(
            self,
            test_size: float = 0.1,
            seed: int = 42,
            train_output_file: Optional[str] = None,
            test_output_file: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.test_size = test_size
        self.seed = seed
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        LOG.info(f"Initializing {self.__class__.__name__} with test_size: {test_size}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RerankerTrainTestSplitter 算子用于分割训练集和测试集。\n\n"
                "功能特点：\n"
                "- 支持按比例随机分割\n"
                "- 自动保存为 JSONL 格式\n"
                "- 测试集包含负样本用于评估\n\n"
                "输出：包含 'split' 字段标记的数据列表"
            )
        else:
            return (
                "RerankerTrainTestSplitter splits data into train/test sets.\n\n"
                "Features:\n"
                "- Random split by proportion\n"
                "- Auto-save as JSONL format\n"
                "- Test set includes negatives for evaluation"
            )

    def forward_batch_input(self, data: List[dict]) -> List[dict]:
        """
        Split the data into train and test sets.

        Args:
            data: List of dict with training samples

        Returns:
            List of dict with 'split' field indicating train/test
        """
        assert isinstance(data, list), "Input data must be a list"
        records = list(data)

        LOG.info(f"Splitting {len(records)} samples with test_size={self.test_size}")

        # Shuffle and split
        random.seed(self.seed)
        shuffled = records.copy()
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
                    # For eval data, rename pos to corpus for compatibility
                    item_copy = {
                        'query': item.get('query', ''),
                        'corpus': item.get('pos', []),
                        'neg': item.get('neg', [])
                    }
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f"Saved test data to {output_path}")

        return train_data + test_data
