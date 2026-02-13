from lazyllm import pipeline
from lazyllm.tools.data import demo1, demo2

def build_demo_pipeline(input_key='text'):
    """构建演示用数据处理流水线（Pipeline），包含若干示例算子并展示如何在 pipeline 上组合使用这些算子。

Args:
    input_key (str): 要处理的文本字段名，默认 'text'

**Returns:**

    一个可调用的 pipeline 对象，调用时会按顺序执行其中注册的算子。


Examples:
    ```python
    from lazyllm.tools.data.pipelines.demo_pipelines import build_demo_pipeline

    ppl = build_demo_pipeline(input_key='text')
    data = [{'text': 'lazyLLM'}]
    res = ppl(data)
    print(res)  # demonstrates how operators are combined and applied
    ```
    """
    with pipeline() as ppl:
        ppl.build_pre_suffix = demo1.build_pre_suffix(input_key=input_key, prefix='Hello, ', suffix='!')
        ppl.process_uppercase = demo1.process_uppercase(input_key=input_key)
        ppl.add_suffix = demo2.AddSuffix(input_key=input_key, suffix='!!!', _max_workers=4)
        ppl.rich_content = demo2.rich_content(input_key=input_key, _concurrency_mode='single')
    return ppl
