import json
import os
from typing import Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.kbcleaning import KnowledgeCleanerPrompt

# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadRAWChunkFile(kbc):
    """加载原始分块文件算子。

该算子从指定路径加载包含原始分块（raw_chunk）的JSON或JSONL文件。
用于知识库清洗流程中加载需要清洗的原始分块数据。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含原始分块数据的数据：
    - _chunks_data: 原始分块数据列表
    - _chunk_path: 分块文件路径


Examples:
    ```python
    from lazyllm.tools.data import kbc

    loader = kbc.KBCLoadRAWChunkFile()

    data = {'chunk_path': '/path/to/raw_chunks.json'}
    result = loader(data)
    # Returns: {'chunk_path': '/path/to/raw_chunks.json', '_chunks_data': [{'raw_chunk': '...'}], '_chunk_path': '/path/to/raw_chunks.json'}
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(
        self,
        data: dict,
        input_key: str = 'chunk_path',
        **kwargs
    ) -> dict:
        chunk_path = data.get(input_key, '')
        if not chunk_path or not os.path.exists(chunk_path):
            LOG.warning(f'Invalid chunk path: {chunk_path}')
            return {**data, '_chunks_data': [], '_chunk_path': chunk_path}

        try:
            if chunk_path.endswith('.json'):
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
            elif chunk_path.endswith('.jsonl'):
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    file_data = [json.loads(line) for line in f]
            else:
                LOG.warning(f'Unsupported file format: {chunk_path}')
                return {**data, '_chunks_data': [], '_chunk_path': chunk_path}

            if not file_data or 'raw_chunk' not in file_data[0]:
                LOG.warning(f"'raw_chunk' field not found in: {chunk_path}")
                return {**data, '_chunks_data': [], '_chunk_path': chunk_path}

            return {**data, '_chunks_data': file_data, '_chunk_path': chunk_path}

        except Exception as e:
            LOG.error(f'Error loading chunk file {chunk_path}: {e}')
            return {**data, '_chunks_data': [], '_chunk_path': chunk_path}


class KBCGenerateCleanedText(kbc):
    """生成清洗后文本的算子。

该算子使用LLM对原始分块文本进行清洗，去除噪声、格式化内容。
支持多语言，当LLM调用失败时会使用原始文本作为回退。

Args:
    llm: LLM服务实例，用于清洗文本。
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含清洗结果的数据：
    - _cleaned_results: 清洗结果列表，每个包含 response、raw_chunk 和 original_item


Examples:
    ```python
    from lazyllm.tools.data import kbc

    # Assuming llm is an LLM service instance
    cleaner = kbc.KBCGenerateCleanedText(llm=llm, lang='en')

    data = {'_chunks_data': [{'raw_chunk': 'Noisy text with errors...'}]}
    result = cleaner(data)
    # Returns: {'_chunks_data': [...], '_cleaned_results': [{'response': 'Cleaned text', 'raw_chunk': '...', 'original_item': {...}}]}
    ```
    """
    def __init__(self, llm=None, lang: str = 'en', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompts = KnowledgeCleanerPrompt(lang=lang)
        if llm is not None:
            # Note: KnowledgeCleanerPrompt may not have system prompt, use empty string
            system_prompt = getattr(self.prompts, 'build_system_prompt', lambda: '')()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        chunks_data = data.get('_chunks_data', [])
        if not chunks_data:
            return {**data, '_cleaned_results': []}

        cleaned_results = []
        for item in chunks_data:
            raw_chunk = item.get('raw_chunk', '')
            if not raw_chunk:
                continue

            # Build prompt for this chunk
            user_prompt = self.prompts.build_prompt(raw_chunk)

            try:
                # Call LLM (system prompt and formatter already set in __init__)
                response = self._llm_serve(user_prompt)

                cleaned_results.append({
                    'response': response,
                    'raw_chunk': raw_chunk,
                    'original_item': item
                })
            except Exception as e:
                LOG.warning(f'Failed to clean text: {e}')
                # Use raw chunk as fallback
                cleaned_results.append({
                    'response': raw_chunk,
                    'raw_chunk': raw_chunk,
                    'original_item': item
                })

        return {**data, '_cleaned_results': cleaned_results}


@data_register('data.kbc', rewrite_func='forward', _concurrency_mode='process')
def extract_cleaned_content(data: dict) -> dict:
    """提取清洗内容函数。

该函数从LLM清洗结果中提取清洗后的文本内容，处理不同的响应格式。
支持从标签 <cleaned_start> 和 <cleaned_end> 之间提取内容。

Args:
    data (dict): 包含清洗结果的数据。

Returns:
    dict: 包含提取后清洗内容的数据：
    - _cleaned_chunks: 清洗后的分块列表，每个包含 raw_chunk、cleaned_chunk 和 original_item


Examples:
    ```python
    from lazyllm.tools.data.operators.knowledge_cleaning.kbc_text_cleaner_batch import extract_cleaned_content

    data = {'_cleaned_results': [{'response': '<cleaned_start>Clean text<cleaned_end>', 'raw_chunk': 'raw', 'original_item': {}}]}
    result = extract_cleaned_content(data)
    # Returns: {'_cleaned_results': [...], '_cleaned_chunks': [{'raw_chunk': 'raw', 'cleaned_chunk': 'Clean text', 'original_item': {}}]}
    ```
    """
    cleaned_results = data.get('_cleaned_results', [])
    if not cleaned_results:
        return {**data, '_cleaned_chunks': []}

    cleaned_chunks = []
    for result in cleaned_results:
        response = result.get('response', '')
        raw_chunk = result.get('raw_chunk', '')
        original_item = result.get('original_item', {})

        # Handle different response types from JsonFormatter
        if isinstance(response, dict):
            # JsonFormatter returned a dict, extract text field or convert to string
            text = response.get('text', '') or response.get('content', '') or str(response)
        elif isinstance(response, list):
            # JsonFormatter returned a list, join or take first item
            text = response[0] if response else ''
            if isinstance(text, dict):
                text = text.get('text', '') or text.get('content', '') or str(text)
        elif isinstance(response, str):
            # JsonFormatter failed to parse, use as-is
            text = response
        else:
            text = str(response)

        # Extract content between tags
        if '<cleaned_start>' in text and '<cleaned_end>' in text:
            try:
                cleaned_text = text.split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
            except IndexError:
                cleaned_text = text.strip()
        else:
            cleaned_text = text.strip()

        cleaned_chunks.append({
            'raw_chunk': raw_chunk,
            'cleaned_chunk': cleaned_text,
            'original_item': original_item
        })

    return {**data, '_cleaned_chunks': cleaned_chunks}


def _clean_save_result(result: dict, output_key: str, output_path: str = '') -> dict:
    result[output_key] = output_path
    for key in ['_chunks_data', '_chunk_path', '_cleaned_results', '_cleaned_chunks']:
        result.pop(key, None)
    return result


def _build_json_items(cleaned_chunks: list) -> list:
    return [{
        'raw_chunk': item['raw_chunk'],
        'cleaned_chunk': item['cleaned_chunk']
    } for item in cleaned_chunks]


def _get_save_output_path(chunk_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        abs_chunk_path = os.path.abspath(chunk_path)
        abs_cwd = os.path.abspath(os.getcwd())
        if abs_chunk_path.startswith(abs_cwd):
            rel_path = os.path.relpath(abs_chunk_path, abs_cwd)
        else:
            rel_path = abs_chunk_path.lstrip('/')

        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        base, ext = os.path.splitext(chunk_path)
        if base.endswith('_cleaned'):
            counter = 1
            output_path = f'{base}_v{counter}{ext}'
            while os.path.exists(output_path):
                counter += 1
                output_path = f'{base}_v{counter}{ext}'
        else:
            output_path = f'{base}_cleaned{ext}'
    return output_path


class KBCSaveCleaned(kbc):
    """保存清洗后数据算子。

该算子将清洗后的分块数据保存为JSON文件，保留原始分块和清洗后分块的对应关系。
支持指定输出目录，会保留原始文件的相对路径结构。

Args:
    output_dir (str, optional): 输出目录路径，默认为 None（保存到原文件所在目录）。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含保存结果的数据：
    - cleaned_chunk_path: 清洗后的分块文件路径


Examples:
    ```python
    from lazyllm.tools.data import kbc

    saver = kbc.KBCSaveCleaned(output_dir='./cleaned_output')

    data = {'_chunk_path': '/path/to/raw_chunks.json', '_cleaned_chunks': [{'raw_chunk': 'raw', 'cleaned_chunk': 'cleaned'}]}
    result = saver(data, output_key='cleaned_chunk_path')
    # Returns: {'cleaned_chunk_path': './cleaned_output/path/to/raw_chunks_cleaned.json'}
    ```
    """
    def __init__(self, output_dir: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_dir = output_dir

    def forward(self, data: dict, output_key: str = 'cleaned_chunk_path', **kwargs) -> dict:
        cleaned_chunks = data.get('_cleaned_chunks', [])
        chunk_path = data.get('_chunk_path', '')
        result = data.copy()

        if not chunk_path:
            return _clean_save_result(result, output_key)

        if not cleaned_chunks:
            LOG.warning(f'No cleaned chunks to save for {chunk_path}')
            return _clean_save_result(result, output_key, chunk_path)

        try:
            json_items = _build_json_items(cleaned_chunks)
            output_path = _get_save_output_path(chunk_path, self.output_dir)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_items, f, ensure_ascii=False, indent=4)

            LOG.info(f'Successfully saved cleaned chunks to {output_path}')
            return _clean_save_result(result, output_key, output_path)

        except Exception as e:
            LOG.error(f'Error saving cleaned chunks: {e}')
            return _clean_save_result(result, output_key)
