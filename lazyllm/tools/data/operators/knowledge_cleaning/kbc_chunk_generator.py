from typing import List
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register


# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCExpandChunks(kbc):
    """将分块文本展开为独立记录的算子。

该算子将包含多个文本分块的数据记录展开为多个独立的数据记录，每个记录包含一个分块。
适用于需要将分块后的文本作为独立样本进行后续处理的场景。

Args:
    output_key (str): 输出字段名，用于存储分块文本，默认为 'raw_chunk'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 展开后的独立数据记录列表，每个记录包含一个分块。


Examples:
    ```python
    from lazyllm.tools.data import kbc

    expander = kbc.KBCExpandChunks(output_key='raw_chunk')

    data = {'text_path': '/path/to/doc.txt', '_chunks': ['chunk1 content', 'chunk2 content', 'chunk3 content']}
    result = expander(data)
    # Returns: [
    #   {'text_path': '/path/to/doc.txt', 'raw_chunk': 'chunk1 content'},
    #   {'text_path': '/path/to/doc.txt', 'raw_chunk': 'chunk2 content'},
    #   {'text_path': '/path/to/doc.txt', 'raw_chunk': 'chunk3 content'}
    # ]
    ```
    """
    def __init__(self, output_key: str = 'raw_chunk', **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.output_key = output_key

    def forward(
        self,
        data: dict,
        **kwargs,
    ) -> List[dict]:
        chunks = data.get('_chunks', [])

        if not chunks:
            return []

        new_records = []
        for chunk_text in chunks:
            new_row = data.copy()
            new_row[self.output_key] = chunk_text
            new_row.pop('_text_content', None)
            new_row.pop('_chunks', None)
            new_records.append(new_row)

        return new_records
