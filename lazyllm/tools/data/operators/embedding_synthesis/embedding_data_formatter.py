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
    """将数据格式化为 FlagEmbedding 训练格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）格式化为 FlagEmbedding 框架所需的训练数据格式。
支持添加指令（instruction）字段用于有监督的 Embedding 训练。

Args:
    instruction (str, optional): 指令文本，用于有监督训练场景。默认为 None。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含 query、pos、neg 和可选 prompt 字段的字典。


Examples:
    ```python
    from lazyllm.tools.data import embedding

    op = embedding.EmbeddingFormatFlagEmbedding(instruction='Represent this sentence for searching relevant passages:')
    result = op({'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe']})
    # Returns: {'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe'], 'prompt': 'Represent this sentence for searching relevant passages:'}
    ```
    """
    def __init__(self, instruction: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.instruction = instruction

    def forward(self, data: dict) -> dict:
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
            'query': query,
            'pos': pos,
            'neg': neg,
        }
        if self.instruction:
            result['prompt'] = self.instruction

        return result


class EmbeddingFormatSentenceTransformers(embedding):
    """将数据格式化为 SentenceTransformers 三元组训练格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）转换为 SentenceTransformers 框架所需的 anchor-positive-negative 三元组格式。
适用于 MultipleNegativesRankingLoss 等损失函数的训练。

Args:
    **kwargs (dict): 可选的参数，传递给父类。

Returns:
    List[dict]: 包含 anchor、positive、negative 字段的字典列表，每对正负样本生成一个三元组。


Examples:
    ```python
    from lazyllm.tools.data import embedding

    op = embedding.EmbeddingFormatSentenceTransformers()
    result = op({'query': 'machine learning', 'pos': ['ML basics'], 'neg': ['cooking tips']})
    # Returns: [{'anchor': 'machine learning', 'positive': 'ML basics', 'negative': 'cooking tips'}]
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict) -> List[dict]:
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
                results.append(
                    {
                        'anchor': query,
                        'positive': p,
                        'negative': n,
                    }
                )

        return results


class EmbeddingFormatTriplet(embedding):
    """将数据格式化为通用三元组格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）转换为标准的三元组格式，
字段名为 query、positive、negative。适用于多种 Embedding 训练框架。

Args:
    **kwargs (dict): 可选的参数，传递给父类。

Returns:
    List[dict]: 包含 query、positive、negative 字段的字典列表，每对正负样本生成一个三元组。


Examples:
    ```python
    from lazyllm.tools.data import embedding

    op = embedding.EmbeddingFormatTriplet()
    result = op({'query': 'deep learning', 'pos': ['neural networks', 'AI'], 'neg': ['history', 'geography']})
    # Returns list of triplets combining each positive with each negative
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict) -> List[dict]:
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
                results.append(
                    {
                        'query': query,
                        'positive': p,
                        'negative': n,
                    }
                )

        return results


class EmbeddingTrainTestSplitter(embedding):
    """将数据集分割为训练集和测试集的算子。

该算子对输入数据进行随机打乱，并按指定比例分割为训练集和测试集。
支持保存分割后的数据到 JSONL 文件，并可按指定键进行分层抽样。

Args:
    test_size (float): 测试集比例，默认为 0.1（即 10%）。
    seed (int): 随机种子，用于可复现的分割结果，默认为 42。
    stratify_key (str, optional): 分层抽样的键名，默认为 None。
    train_output_file (str, optional): 训练集输出文件路径，默认为 None。
    test_output_file (str, optional): 测试集输出文件路径，默认为 None。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 包含训练集和测试集的所有样本，每个样本添加了 'split' 字段标记所属集合。


Examples:
    ```python
    from lazyllm.tools.data import embedding

    op = embedding.EmbeddingTrainTestSplitter(test_size=0.2, seed=123, train_output_file='train.jsonl', test_output_file='test.jsonl')
    data = [{'query': 'q1', 'pos': 'p1'}, {'query': 'q2', 'pos': 'p2'}, {'query': 'q3', 'pos': 'p3'}]
    result = op(data)
    # Returns all samples with 'split' field ('train' or 'test')
    # Saves train data to train.jsonl and test data to test.jsonl
    ```
    """
    def __init__(
        self,
        test_size: float = 0.1,
        seed: int = 42,
        stratify_key: Optional[str] = None,
        train_output_file: Optional[str] = None,
        test_output_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.test_size = test_size
        self.seed = seed
        self.stratify_key = stratify_key
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        LOG.info(
            f'Initializing {self.__class__.__name__} with test_size: {test_size}'
        )

    def forward_batch_input(
        self,
        inputs: List[dict],
        **kwargs,
    ) -> List[dict]:
        assert isinstance(inputs, list), 'inputs must be a list of dict'

        LOG.info(
            f'Splitting {len(inputs)} samples with test_size={self.test_size}'
        )

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

        LOG.info(
            f'Split completed: {len(train_data)} train, {len(test_data)} test'
        )

        if self.train_output_file:
            output_path = Path(self.train_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in train_data:
                    item_copy = {
                        k: v for k, v in item.items() if k != 'split'
                    }
                    f.write(
                        json.dumps(item_copy, ensure_ascii=False) + '\n'
                    )
            LOG.info(f'Saved train data to {output_path}')

        if self.test_output_file:
            output_path = Path(self.test_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in test_data:
                    item_copy = {
                        k: v for k, v in item.items() if k != 'split'
                    }
                    f.write(
                        json.dumps(item_copy, ensure_ascii=False) + '\n'
                    )
            LOG.info(f'Saved test data to {output_path}')

        return train_data + test_data
