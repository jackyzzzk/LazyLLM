from collections import defaultdict
import json
import os
import tempfile
from typing import Dict
import lazyllm

class FileHandlerBase:
    """FileHandlerBase 是处理微调数据文件的基类，主要用于验证和转换微调数据格式。  
该类不支持直接实例化，需要子类继承并实现特定的文件格式转换逻辑。

功能包括：
    1. 验证微调数据文件格式是否为标准的 `.jsonl`。
    2. 检查每条数据是否包含符合规范的消息格式（包含 `role` 和 `content` 字段）。
    3. 验证角色类型是否在允许范围内（system、knowledge、user、assistant）。
    4. 确保每个对话示例包含至少一条 assistant 回复。
    5. 提供临时文件存储机制，便于后续处理。


Examples:
    >>> import lazyllm
    >>> from lazyllm.module.llms.onlinemodule.fileHandler import FileHandlerBase
    >>> import tempfile
    >>> import json
    >>> sample_data = [
    ...     {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
    ...     {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}]}
    ... ] 
    >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    ...     for item in sample_data:
    ...         f.write(json.dumps(item, ensure_ascii=False) + '
    ')
    ...     temp_file_path = f.name
    >>> class CustomFileHandler(FileHandlerBase):
    ...     def _convert_file_format(self, filepath: str) -> str:
    ...         with open(filepath, 'r', encoding='utf-8') as f:
    ...             data = [json.loads(line) for line in f]
    ...         converted_data = []
    ...         for item in data:
    ...             messages = item.get('messages', [])
    ...             conversation = []
    ...             for msg in messages:
    ...                 conversation.append(f"{msg['role']}: {msg['content']}")
    ...             converted_data.append('
    '.join(conversation))
    ...         return '
    ---
    '.join(converted_data)
    >>> handler = CustomFileHandler()
    >>> try:
    ...     result = handler.get_finetune_data(temp_file_path)
    ...     print("数据验证和转换成功")
    ... except Exception as e:
    ...     print(f"错误: {e}")
    ... finally:
    ...     import os
    ...     os.unlink(temp_file_path)
    """

    def __init__(self):
        self._roles = ['system', 'knowledge', 'user', 'assistant']

    def _validate_json(self, data_path: str) -> None:  # noqa C901
        # Check if file name format
        if os.path.splitext(data_path)[-1] != '.jsonl':
            raise ValueError('The file name must end with .jsonl')
        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'File {data_path} does not exist.')

        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        # Initial dataset stats
        lazyllm.LOG.info('Num examples:', len(dataset))
        lazyllm.LOG.info('First example:')
        for message in dataset[0]['messages']:
            lazyllm.LOG.info(message)

        # Format error checks
        format_error: Dict[str, list[int]] = defaultdict(list)
        for index, line in enumerate(dataset, start=1):
            # Check if example is a dictionary type
            if not isinstance(line, dict):
                format_error['data_type'].append(index)
                continue

            messages = line.get('messages', None)
            # Check if messages keyword exists
            if messages is None:
                format_error['missing_messages_list'].append(index)
                continue

            for message in messages:
                if 'role' not in message or 'content' not in message:
                    format_error['message_missing_key'].append(index)

                if any(k not in ('role', 'content') for k in message):
                    format_error['message_unrecognized_key'].append(index)

                if message.get('role', None) not in self._roles:
                    format_error['unrecognized_role'].append(index)

                content = message.get('content', None)
                if content is None or not isinstance(content, str):
                    format_error['missing_content'].append(index)

            if not any(message.get('role', None) == 'assistant' for message in messages):
                format_error['example_missing_assistant_message'].append(index)

        if format_error:
            lazyllm.LOG.error('Found errors: ')
            for k, v in format_error.items():
                lazyllm.LOG.error(f'Error Type: {k}, Error number: {len(v)}')
                lazyllm.LOG.error(f'Error Type: {k}, Error line number: {v}')
        else:
            lazyllm.LOG.info('No errors found')

    def get_finetune_data(self, filepath: str) -> str:
        """获取并处理微调数据文件，包括验证文件格式和转换为目标平台支持的格式。

Args:
    filepath (str): 微调数据文件的路径，必须是.jsonl格式
"""
        self._validate_json(filepath)
        self._save_tempfile(self._convert_file_format(filepath))

    def _save_tempfile(self, data: str):
        self._dataHandler = tempfile.TemporaryFile()
        self._dataHandler.write(data.encode())
        self._dataHandler.seek(0)

    def _convert_file_format(self, filepath: str) -> str:
        raise NotImplementedError
