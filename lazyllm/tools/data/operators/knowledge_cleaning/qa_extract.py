import json
from pathlib import Path
from typing import List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadQAData(kbc):
    """加载问答数据的算子。

该算子从输入数据或分块文件中加载问答数据。首先检查输入数据中是否已包含问答数据，
如果没有则尝试从增强分块文件、清洗后分块文件或普通分块文件中加载。

Args:
    qa_key (str): 问答数据字段名，默认为 'QA_pairs'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含问答数据的数据：
    - _qa_data: 加载的问答数据
    - _source_file: 数据来源文件路径（如果从文件加载）


Examples:
    ```python
    from lazyllm.tools.data import kbc

    loader = kbc.KBCLoadQAData(qa_key='QA_pairs')

    # From existing data
    data = {'QA_pairs': [{'question': 'Q1', 'answer': 'A1'}]}
    result = loader(data)
    # Returns: {'QA_pairs': [...], '_qa_data': [...]}

    # From file
    data = {'enhanced_chunk_path': '/path/to/enhanced.json'}
    result = loader(data)
    # Returns: {'enhanced_chunk_path': '...', '_qa_data': [...], '_source_file': '/path/to/enhanced.json'}
    ```
    """
    def __init__(self, qa_key: str = 'QA_pairs', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.qa_key = qa_key

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        # Check if QA data already exists in the data
        if self.qa_key in data:
            return {**data, '_qa_data': data.get(self.qa_key)}

        # Try to load from chunk files
        path_keys = ['enhanced_chunk_path', 'cleaned_chunk_path', 'chunk_path']

        for path_key in path_keys:
            file_path = data.get(path_key)
            if not file_path or not Path(file_path).exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    chunks = chunks if isinstance(chunks, list) else [chunks]

                    for chunk in chunks:
                        if self.qa_key in chunk:
                            return {
                                **data,
                                '_qa_data': chunk[self.qa_key],
                                '_source_file': file_path
                            }
            except Exception as e:
                LOG.error(f'Failed to load {file_path}: {e}')
                continue

        # No QA data found
        return {**data, '_qa_data': None}


class KBCExtractQAPairs(kbc):
    """提取问答对的算子。

该算子从加载的问答数据中提取问答对，并将其转换为标准格式。
支持自定义指令、问题和答案的输出字段名。

Args:
    qa_key (str): 问答数据字段名，默认为 'QA_pairs'。
    instruction (str): 指令文本，默认为 'Please answer the following question based on the provided information.'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 提取的问答对列表，每个包含 instruction、input 和 output 字段。


Examples:
    ```python
    from lazyllm.tools.data import kbc

    extractor = kbc.KBCExtractQAPairs(
        qa_key='QA_pairs',
        instruction='Please answer based on the context.'
    )

    data = {'_qa_data': {'qa_pairs': [{'question': 'What is AI?', 'answer': 'Artificial Intelligence'}]}}
    result = extractor(
        data,
        output_instruction_key='instruction',
        output_question_key='input',
        output_answer_key='output'
    )
    # Returns: [{'instruction': 'Please answer based on the context.', 'input': 'What is AI?', 'output': 'Artificial Intelligence'}]
    ```
    """
    def __init__(
        self,
        qa_key: str = 'QA_pairs',
        instruction: str = 'Please answer the following question based on the provided information.',
        **kwargs
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.qa_key = qa_key
        self.instruction = instruction

    def forward(
        self,
        data: dict,
        output_instruction_key: str = 'instruction',
        output_question_key: str = 'input',
        output_answer_key: str = 'output',
        **kwargs
    ) -> List[dict]:
        qa_data = data.get('_qa_data')
        if not qa_data:
            return []

        # Extract qa_pairs - handle both dict with 'qa_pairs' key and direct list
        qa_list = qa_data.get('qa_pairs', []) if isinstance(qa_data, dict) else qa_data
        if not isinstance(qa_list, list):
            qa_list = [qa_list] if isinstance(qa_list, dict) else []

        results = []
        for qa in qa_list:
            if not isinstance(qa, dict):
                continue

            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()

            if not question or not answer:
                continue

            item = {
                output_instruction_key: self.instruction,
                output_question_key: question,
                output_answer_key: answer
            }
            results.append(item)

        return results
