from ..base_data import data_register


Demo1 = data_register.new_group('demo1')
Demo2 = data_register.new_group('demo2')

@data_register('data.demo1', rewrite_func='forward_batch_input')
def build_pre_suffix(data, input_key='content', prefix='', suffix=''):
    """对输入列表中每项在指定字段前后添加前缀和后缀。此算子以批处理函数注册（forward_batch_input）。

Args:
    data (list[dict]): 输入列表
    input_key (str): 文本字段名
    prefix (str): 要添加的前缀
    suffix (str): 要添加的后缀


Examples:
    ```python
    from lazyllm.tools.data.operators.demo_ops import build_pre_suffix

    op = build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
    print(op([{'text': 'world'}]))
    # [{'text': 'Hello, world!'}]
    ```
    """
    assert isinstance(data, list)
    for item in data:
        item[input_key] = f'{prefix}{item.get(input_key, "")}{suffix}'
    return data

@data_register('data.demo1', rewrite_func='forward', _concurrency_mode='process')
def process_uppercase(data, input_key='content'):
    """将输入文本字段转换为大写。适用于单条处理函数注册（forward）。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'


Examples:
    ```python
    from lazyllm.tools.data.operators.demo_ops import process_uppercase

    op = process_uppercase(input_key='text')
    print(op({'text': 'hello'}))  # {'text': 'HELLO'}
    ```
    """
    assert isinstance(data, dict)
    data[input_key] = data.get(input_key, '').upper()
    return data

class AddSuffix(Demo2):
    """通过类方式实现的算子，为指定字段添加后缀。支持并发配置（通过构造参数）。

Args:
    suffix (str): 要添加的后缀
    input_key (str): 文本字段名
    _max_workers (int|None): 可选，最大并发数
    _concurrency_mode (str): 可选，并发模式
    _save_data (bool): 可选，是否保存结果


Examples:
    ```python
    from lazyllm.tools.data.operators.demo_ops import AddSuffix

    op = AddSuffix(suffix='!!!', input_key='text', _max_workers=2)
    print(op([{'text': 'wow'}]))  # [{'text': 'wow!!!'}]
    ```
    """
    def __init__(self, suffix, input_key='content', _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.suffix = suffix
        self.input_key = input_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        data[self.input_key] = f'{data.get(self.input_key, "")}{self.suffix}'
        return data

@data_register('data.demo2', rewrite_func='forward', _concurrency_mode='process')
def rich_content(data, input_key='content'):
    """将单条输入拆分为多条输出，生成富内容表示（原始 + 若干派生）。适用于返回 list 的 forward。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名


Examples:
    ```python
    from lazyllm.tools.data.operators.demo_ops import rich_content

    op = rich_content(input_key='text')
    print(op({'text': 'This is a test.'}))
    # [
    #   {'text': 'This is a test.'},
    #   {'text': 'This is a test. - part 1'},
    #   {'text': 'This is a test. - part 2'}
    # ]
    ```
    """
    assert isinstance(data, dict)
    content = data.get(input_key, '')
    new_res = [data]
    for i in range(2):
        new_data = data.copy()
        new_data[input_key] = f'{content} - part {i+1}'
        new_res.append(new_data)
    return new_res

@data_register('data.demo2', rewrite_func='forward')
def error_prone_op(data, input_key='content'):
    """一个用于测试的算子：在特定输入（content == 'fail'）时抛出异常，否则返回处理后的字典结果。用于验证错误收集与跳过逻辑。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名


Examples:
    ```python
    from lazyllm.tools.data.operators.demo_ops import error_prone_op

    op = error_prone_op(input_key='text', _save_data=True, _concurrency_mode='single')
    res = op([{'text': 'ok'}, {'text': 'fail'}, {'text': 'ok2'}])
    # valid results skip the failed item; error details written to error file
    ```
    """
    assert isinstance(data, dict)
    content = data.get(input_key, '')
    if content == 'fail':
        raise ValueError('Intentional error for testing.')
    data[input_key] = f'Processed: {content}'
    return data
