import json
import os
from typing import Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.text2qa import Text2MultiHopQAGeneratorPrompt


# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadChunkFile(kbc):
    """加载分块文件算子。

该算子从指定路径加载JSON或JSONL格式的分块文件。
支持从知识库清洗流程中生成的分块结果文件。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含分块数据的数据：
    - _chunks_data: 分块数据列表
    - _chunk_path: 分块文件路径


Examples:
    ```python
    from lazyllm.tools.data import kbc

    loader = kbc.KBCLoadChunkFile()

    data = {'chunk_path': '/path/to/chunks.json'}
    result = loader(data)
    # Returns: {'chunk_path': '/path/to/chunks.json', '_chunks_data': [...], '_chunk_path': '/path/to/chunks.json'}
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(
        self,
        data: dict,
        input_key: str = 'chunk_path',
        **kwargs,
    ) -> dict:
        import os

        chunk_path = data.get(input_key, '')

        if not chunk_path or not os.path.exists(chunk_path):
            LOG.warning(f'Invalid chunk path: {chunk_path}')
            return {**data, '_chunks_data': []}

        try:
            if str(chunk_path).endswith('.json'):
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
            elif str(chunk_path).endswith('.jsonl'):
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    file_data = [json.loads(line) for line in f]
            else:
                LOG.warning(f'Unsupported file format: {chunk_path}')
                return {**data, '_chunks_data': []}

            return {
                **data,
                '_chunks_data': file_data,
                '_chunk_path': chunk_path,
            }

        except Exception as e:
            LOG.error(f'Error loading chunk file {chunk_path}: {e}')
            return {**data, '_chunks_data': []}


class KBCPreprocessText(kbc):
    """文本预处理算子。

该算子对加载的分块文本进行预处理，根据长度过滤分块。
只保留长度在指定范围内的分块，避免处理过短或过长的文本。

Args:
    min_length (int): 最小文本长度，默认为 100。
    max_length (int): 最大文本长度，默认为 200000。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含预处理结果的数据：
    - _processed_chunks: 预处理后的分块列表


Examples:
    ```python
    from lazyllm.tools.data import kbc

    processor = kbc.KBCPreprocessText(min_length=50, max_length=10000)

    data = {'_chunks_data': [{'cleaned_chunk': 'Short text.'}, {'cleaned_chunk': 'A much longer text that meets the length requirements and will be processed.'}]}
    result = processor(data, text_field='cleaned_chunk')
    # Returns: {'_chunks_data': [...], '_processed_chunks': [{'text': 'A much longer text...', 'original_data': {...}}]}
    ```
    """
    def __init__(self, min_length: int = 100, max_length: int = 200000, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def forward(
        self,
        data: dict,
        text_field: str = 'cleaned_chunk',
        **kwargs,
    ) -> dict:
        chunks_data = data.get('_chunks_data', [])
        if not chunks_data:
            return {**data, '_processed_chunks': []}

        processed = []
        for item in chunks_data:
            text = item.get(text_field, '')
            if not isinstance(text, str):
                continue

            text = text.strip()
            if self.min_length <= len(text) <= self.max_length:
                processed.append(
                    {
                        'text': text,
                        'original_data': item,
                    }
                )

        return {**data, '_processed_chunks': processed}


class KBCExtractInfoPairs(kbc):
    """信息对提取算子。

该算子从预处理后的文本中提取信息对，用于生成多跳问答。
根据语言类型（中文或英文）使用不同的句子分割符，
提取前提-中间-结论三元组和相关上下文。

Args:
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含信息对的数据：
    - _info_pairs: 信息对列表，每个包含 premise、intermediate、conclusion 和 related_contexts


Examples:
    ```python
    from lazyllm.tools.data import kbc

    extractor = kbc.KBCExtractInfoPairs(lang='en')

    data = {'_processed_chunks': [{'text': 'First sentence. Second sentence. Third sentence.', 'original_data': {}}]}
    result = extractor(data)
    # Returns: {'_processed_chunks': [...], '_info_pairs': [{'premise': 'First sentence', 'intermediate': 'Second sentence', 'conclusion': 'Third sentence', 'related_contexts': [], 'original_data': {}}]}
    ```
    """
    def __init__(self, lang: str = 'en', **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.lang = lang

    def forward(self, data: dict, **kwargs) -> dict:
        processed_chunks = data.get('_processed_chunks', [])
        if not processed_chunks:
            return {**data, '_info_pairs': []}

        all_info_pairs = []

        for chunk in processed_chunks:
            text = chunk.get('text', '')
            original_data = chunk.get('original_data', {})

            if self.lang == 'en':
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            else:
                sentences = [s.strip() for s in text.split('。') if s.strip()]

            for i in range(len(sentences) - 2):
                if len(sentences[i]) > 10 and len(sentences[i + 1]) > 10:
                    info_pair = {
                        'premise': sentences[i],
                        'intermediate': sentences[i + 1],
                        'conclusion': (
                            sentences[i + 2]
                            if i + 2 < len(sentences)
                            else ''
                        ),
                        'related_contexts': [
                            s
                            for j, s in enumerate(sentences)
                            if j not in (i, i + 1) and len(s) > 10
                        ][:2],
                        'original_data': original_data,
                    }
                    all_info_pairs.append(info_pair)

        return {**data, '_info_pairs': all_info_pairs}


class KBCGenerateMultiHopQA(kbc):
    """多跳问答生成算子。

该算子使用LLM根据提取的信息对生成多跳问答对。
多跳问答需要多个推理步骤才能回答，适用于训练复杂的问答模型。

Args:
    llm: LLM服务实例，用于生成问答对。
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含生成的问答结果的数据：
    - _qa_results: 问答结果列表，每个包含 response 和 info_pair


Examples:
    ```python
    from lazyllm.tools.data import kbc

    # Assuming llm is an LLM service instance
    generator = kbc.KBCGenerateMultiHopQA(llm=llm, lang='en')

    data = {'_info_pairs': [{'premise': 'A', 'intermediate': 'B', 'conclusion': 'C', 'original_data': {}}]}
    result = generator(data)
    # Returns: {'_info_pairs': [...], '_qa_results': [{'response': {...}, 'info_pair': {...}}]}
    ```
    """
    def __init__(self, llm=None, lang: str = 'en', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        self.prompt_template = Text2MultiHopQAGeneratorPrompt(lang=lang)

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = (
                llm.share()
                .prompt(system_prompt)
                .formatter(JsonFormatter())
            )
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict, **kwargs) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        info_pairs = data.get('_info_pairs', [])
        if not info_pairs:
            return {**data, '_qa_results': []}

        qa_results = []

        for pair in info_pairs:
            # Build context from info pair
            context = (
                f"{pair['premise']}. "
                f"{pair['intermediate']}. "
                f"{pair['conclusion']}"
            )

            # Build prompt for this info pair
            user_prompt = self.prompt_template.build_prompt(context)

            try:
                response = self._llm_serve(user_prompt)

                qa_results.append(
                    {
                        'response': response,
                        'info_pair': pair,
                    }
                )

            except Exception as e:
                LOG.warning(f'Failed to generate QA: {e}')

        return {**data, '_qa_results': qa_results}


@data_register('data.kbc', rewrite_func='forward', _concurrency_mode='process')
def parse_qa_pairs(data: dict) -> dict:
    """解析问答对函数。

该函数解析LLM生成的问答响应，提取有效的问答对。
支持多种响应格式（字典、列表、字符串），并将解析结果与原始数据合并。

Args:
    data (dict): 包含问答结果的数据。

Returns:
    dict: 包含解析后的问答对的数据：
    - _qa_pairs: 解析后的问答对列表


Examples:
    ```python
    from lazyllm.tools.data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch import parse_qa_pairs

    data = {'_qa_results': [{'response': {'question': 'What is AI?', 'answer': 'Artificial Intelligence'}, 'info_pair': {'original_data': {'id': 1}}}]}
    result = parse_qa_pairs(data)
    # Returns: {'_qa_results': [...], '_qa_pairs': [{'id': 1, 'qa_pairs': {'question': 'What is AI?', 'answer': 'Artificial Intelligence'}}]}
    ```
    """
    qa_results = data.get('_qa_results', [])
    if not qa_results:
        return {**data, '_qa_pairs': []}

    all_qa_pairs = []

    for qa_result in qa_results:
        response = qa_result.get('response', '')
        info_pair = qa_result.get('info_pair', {})
        original_data = info_pair.get('original_data', {})

        if isinstance(response, dict):
            if 'question' in response:
                all_qa_pairs.append(
                    {**original_data, 'qa_pairs': response}
                )

        elif isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and 'question' in item:
                    all_qa_pairs.append(
                        {**original_data, 'qa_pairs': item}
                    )

        elif isinstance(response, str):
            LOG.warning(
                f'JsonFormatter failed to parse response, '
                f'skipping: {response[:100]}...'
            )

    return {**data, '_qa_pairs': all_qa_pairs}


def _clean_enhanced_result(result: dict, output_key: str, output_path: str = '') -> dict:
    result[output_key] = output_path
    for key in [
        '_chunks_data', '_chunk_path', '_processed_chunks',
        '_info_pairs', '_qa_results', '_qa_pairs',
    ]:
        result.pop(key, None)
    return result


def _build_enhanced_data(chunks_data: list, qa_pairs: list) -> list:
    enhanced_data = []
    for item in chunks_data:
        enhanced_item = item.copy()
        matching_qa = [
            qa for qa in qa_pairs
            if qa.get('cleaned_chunk') == item.get('cleaned_chunk')
        ]
        if matching_qa:
            enhanced_item['qa_pairs'] = matching_qa[0].get('qa_pairs', {})
        enhanced_data.append(enhanced_item)
    return enhanced_data


def _get_output_path(chunk_path: str, output_dir: Optional[str]) -> str:
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
        if base.endswith('_enhanced'):
            counter = 1
            output_path = f'{base}_v{counter}{ext}'
            while os.path.exists(output_path):
                counter += 1
                output_path = f'{base}_v{counter}{ext}'
        else:
            output_path = f'{base}_enhanced{ext}'
    return output_path


def _save_enhanced_data(enhanced_data: list, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        if str(output_path).endswith('.json'):
            json.dump(enhanced_data, f, ensure_ascii=False, indent=4)
        else:
            for item in enhanced_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class KBCSaveEnhanced(kbc):
    """保存增强数据算子。

该算子将生成的问答对与原始分块数据合并，保存为增强后的分块文件。
支持指定输出目录，会保留原始文件的相对路径结构。

Args:
    output_dir (str, optional): 输出目录路径，默认为 None（保存到原文件所在目录）。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含保存结果的数据：
    - enhanced_chunk_path: 增强后的分块文件路径


Examples:
    ```python
    from lazyllm.tools.data import kbc

    saver = kbc.KBCSaveEnhanced(output_dir='./enhanced_output')

    data = {'_chunk_path': '/path/to/chunks.json', '_chunks_data': [{'id': 1, 'text': 'chunk1'}], '_qa_pairs': [{'id': 1, 'qa_pairs': {'question': 'Q1', 'answer': 'A1'}}]}
    result = saver(data, output_key='enhanced_chunk_path')
    # Returns: {'enhanced_chunk_path': './enhanced_output/path/to/chunks_enhanced.json'}
    ```
    """
    def __init__(self, output_dir: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_dir = output_dir

    def forward(self, data: dict, output_key: str = 'enhanced_chunk_path', **kwargs) -> dict:
        chunk_path = data.get('_chunk_path', '')
        result = data.copy()

        if not chunk_path:
            return _clean_enhanced_result(result, output_key)

        try:
            enhanced_data = _build_enhanced_data(
                data.get('_chunks_data', []),
                data.get('_qa_pairs', [])
            )
            output_path = _get_output_path(chunk_path, self.output_dir)
            _save_enhanced_data(enhanced_data, output_path)
            LOG.info(f'Saved enhanced chunks to {output_path}')
            return _clean_enhanced_result(result, output_key, output_path)
        except Exception as e:
            LOG.error(f'Error saving enhanced chunks: {e}')
            return _clean_enhanced_result(result, output_key)
