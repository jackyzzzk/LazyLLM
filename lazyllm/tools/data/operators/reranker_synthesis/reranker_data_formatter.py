import json
import random
from pathlib import Path
from typing import List, Optional

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register


if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


@data_register('data.reranker', rewrite_func='forward', _concurrency_mode='thread')
def validate_reranker_data(
    data: dict,
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    input_neg_key: str = 'neg',
) -> dict:
    """验证重排序数据的函数。

该函数验证输入数据是否包含必要的字段（query、正样本），并确保正样本和负样本为列表格式。

Args:
    data (dict): 输入数据，应包含 query、pos 和 neg 字段。
    input_query_key (str): 查询字段名，默认为 'query'。
    input_pos_key (str): 正样本字段名，默认为 'pos'。
    input_neg_key (str): 负样本字段名，默认为 'neg'。

Returns:
    dict: 验证后的数据，包含：
    - _is_valid: 数据是否有效
    - _error: 错误信息（如果无效）
    - _query, _pos, _neg: 标准化后的字段值


Examples:
    ```python
    from lazyllm.tools.data.operators.reranker_synthesis.reranker_data_formatter import validate_reranker_data

    data = {'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe']}
    result = validate_reranker_data(data)
    # Returns: {'query': '...', 'pos': [...], 'neg': [...], '_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking recipe']}
    ```
    """
    query = data.get(input_query_key, '')
    pos = data.get(input_pos_key, [])

    if not query:
        return {**data, '_is_valid': False, '_error': 'Missing query'}

    if not pos:
        return {**data, '_is_valid': False, '_error': 'Missing positive samples'}

    # Ensure pos and neg are lists
    if not isinstance(pos, list):
        pos = [pos]

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


class RerankerFormatFlagReranker(reranker):
    """FlagReranker格式转换算子。

该算子将验证后的数据转换为FlagReranker训练格式。确保负样本数量符合训练组大小要求，
如果负样本不足会复制填充，如果过多会截断。

Args:
    train_group_size (int): 训练组大小（包含1个正样本），默认为 8。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 转换后的数据列表，每个包含 query、pos 和 neg 字段。


Examples:
    ```python
    from lazyllm.tools.data import reranker

    formatter = reranker.RerankerFormatFlagReranker(train_group_size=8)

    data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking', 'history']}
    result = formatter(data)
    # Returns: [{'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking', 'history', ...]}]
    ```
    """
    def __init__(self, train_group_size: int = 8, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.train_group_size = train_group_size

    def forward(self, data: dict, **kwargs) -> List[dict]:
        if not data.get('_is_valid'):
            return []

        query = data['_query']
        pos = data['_pos']
        neg = data['_neg']

        # Ensure neg has exactly train_group_size - 1 samples
        num_neg_needed = self.train_group_size - 1
        if len(neg) < num_neg_needed:
            # Pad with duplicates if needed
            neg = (neg * (num_neg_needed // len(neg) + 1))[:num_neg_needed] if neg else []
        else:
            neg = neg[:num_neg_needed]

        return [{
            'query': query,
            'pos': pos,
            'neg': neg,
        }]


class RerankerFormatCrossEncoder(reranker):
    """CrossEncoder格式转换算子。

该算子将验证后的数据转换为CrossEncoder训练格式。每个查询-文档对作为一个独立样本，
正样本标记为1，负样本标记为0。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 转换后的数据列表，每个包含 query、document 和 label 字段。


Examples:
    ```python
    from lazyllm.tools.data import reranker

    formatter = reranker.RerankerFormatCrossEncoder()

    data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
    result = formatter(data)
    # Returns: [{'query': 'machine learning', 'document': 'ML tutorial', 'label': 1}, {'query': 'machine learning', 'document': 'cooking', 'label': 0}]
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(self, data: dict, **kwargs) -> List[dict]:
        if not data.get('_is_valid'):
            return []

        query = data['_query']
        pos = data['_pos']
        neg = data['_neg']

        results = []

        # Positive samples with label 1
        for p in pos:
            results.append({'query': query, 'document': p, 'label': 1})

        # Negative samples with label 0
        for n in neg:
            results.append({'query': query, 'document': n, 'label': 0})

        return results


class RerankerFormatPairwise(reranker):
    """Pairwise格式转换算子。

该算子将验证后的数据转换为Pairwise训练格式。创建正样本和负样本的成对组合，
用于训练排序模型区分相关和不相关文档。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 转换后的数据列表，每个包含 query、doc_pos 和 doc_neg 字段。


Examples:
    ```python
    from lazyllm.tools.data import reranker

    formatter = reranker.RerankerFormatPairwise()

    data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
    result = formatter(data)
    # Returns: [{'query': 'machine learning', 'doc_pos': 'ML tutorial', 'doc_neg': 'cooking'}]
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(self, data: dict, **kwargs) -> List[dict]:
        if not data.get('_is_valid'):
            return []

        query = data['_query']
        pos = data['_pos']
        neg = data['_neg']

        results = []

        # Create pairwise comparisons
        for p in pos:
            for n in neg:
                results.append({'query': query, 'doc_pos': p, 'doc_neg': n})

        return results

class RerankerTrainTestSplitter(reranker):
    """重排序训练集/测试集分割算子。

该算子将数据集随机分割为训练集和测试集，支持指定分割比例和随机种子。
可以保存训练集和测试集到指定文件，测试集会转换格式以兼容评估需求。

Args:
    test_size (float): 测试集比例，默认为 0.1（即10%）。
    seed (int): 随机种子，用于可复现的分割，默认为 42。
    train_output_file (str, optional): 训练集输出文件路径，默认为 None。
    test_output_file (str, optional): 测试集输出文件路径，默认为 None。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 分割后的数据列表，每个样本包含 split 字段标记所属集合（'train' 或 'test'）。


Examples:
    ```python
    from lazyllm.tools.data import reranker

    splitter = reranker.RerankerTrainTestSplitter(
        test_size=0.2,
        seed=123,
        train_output_file='train.jsonl',
        test_output_file='test.jsonl'
    )

    data = [
        {'query': 'q1', 'pos': ['p1'], 'neg': ['n1']},
        {'query': 'q2', 'pos': ['p2'], 'neg': ['n2']}
    ]
    result = splitter(data)
    # Returns: [{'query': 'q1', 'pos': ['p1'], 'neg': ['n1'], 'split': 'train'}, {'query': 'q2', 'pos': ['p2'], 'neg': ['n2'], 'split': 'test'}]
    ```
    """
    def __init__(
            self,
            test_size: float = 0.1,
            seed: int = 42,
            train_output_file: Optional[str] = None,
            test_output_file: Optional[str] = None,
            **kwargs
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.test_size = test_size
        self.seed = seed
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        LOG.info(f'Initializing {self.__class__.__name__} with test_size: {test_size}')

    def forward_batch_input(self, data: List[dict]) -> List[dict]:
        assert isinstance(data, list), 'Input data must be a list'
        records = list(data)

        LOG.info(f'Splitting {len(records)} samples with test_size={self.test_size}')

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

        LOG.info(f'Split completed: {len(train_data)} train, {len(test_data)} test')

        # Save to files if specified
        if self.train_output_file:
            output_path = Path(self.train_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in train_data:
                    item_copy = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f'Saved train data to {output_path}')

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
            LOG.info(f'Saved test data to {output_path}')

        return train_data + test_data
