import random
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


@data_register('data.reranker', rewrite_func='forward', _concurrency_mode='thread')
def validate_reranker_embedding_data(
    data: dict,
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    input_neg_key: str = 'neg',
) -> dict:
    """验证Embedding数据用于重排序的函数。

该函数验证输入的Embedding格式数据是否适合转换为重排序格式。检查query和正样本是否存在，
并确保正样本和负样本为列表格式。

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
    from lazyllm.tools.data.operators.reranker_synthesis.reranker_from_embedding_converter import validate_reranker_embedding_data

    data = {'query': 'machine learning', 'pos': 'ML tutorial', 'neg': ['cooking']}
    result = validate_reranker_embedding_data(data)
    # Returns: {'query': '...', 'pos': 'ML tutorial', 'neg': [...], '_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
    ```
    """
    query = data.get(input_query_key, '')
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
    """调整重排序负样本数量的算子。

该算子调整负样本数量以匹配目标数量。如果负样本过多则截断，如果不足则通过随机采样进行填充。
使用基于查询内容的确定性随机种子以保证可复现性。

Args:
    adjust_neg_count (int): 目标负样本数量，默认为 7。
    seed (int): 随机种子，用于填充时的随机选择，默认为 42。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 调整后的数据，包含更新后的 _neg 字段。


Examples:
    ```python
    from lazyllm.tools.data import reranker

    adjuster = reranker.RerankerAdjustNegatives(adjust_neg_count=5, seed=123)

    # Too many negatives
    data = {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']}
    result = adjuster(data)
    # Returns: {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2', 'n3', 'n4', 'n5']}

    # Too few negatives
    data = {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2']}
    result = adjuster(data)
    # Returns: {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2', 'n1', 'n2', 'n1']}
    ```
    """
    def __init__(self, adjust_neg_count: int = 7, seed: int = 42, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
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
            local_random = random.Random(f'{self.seed}_{data["_query"]}')
            while len(neg) < self.adjust_neg_count:
                neg.append(local_random.choice(neg))

        return {**data, '_neg': neg}


class RerankerBuildFormat(reranker):
    """构建重排序格式的算子。

该算子将验证后的数据转换为标准的重排序训练格式。输出包含 query、pos 和 neg 字段的字典，
不包含提示或指令字段。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 重排序格式的数据，包含 query、pos 和 neg 字段。如果数据无效则返回空字典。


Examples:
    ```python
    from lazyllm.tools.data import reranker

    builder = reranker.RerankerBuildFormat()

    data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
    result = builder(data)
    # Returns: {'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking']}
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(self, data: dict, **kwargs) -> dict:
        if not data.get('_is_valid'):
            return {}

        # Build reranker format (no prompt/instruction)
        reranker_item = {
            'query': data['_query'],
            'pos': data['_pos'],
            'neg': data['_neg'],
        }

        return reranker_item
