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


class KBCGenerateCleanedTextSingle(kbc):
    """单条文本清洗生成算子。

该算子使用LLM对单条原始文本进行清洗，去除噪声、格式化内容。
适用于单条数据的实时清洗场景，当LLM调用失败时会使用原始文本作为回退。

Args:
    llm: LLM服务实例，用于清洗文本。
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含清洗响应的数据：
    - _cleaned_response: LLM的清洗响应


Examples:
    ```python
    from lazyllm.tools.data import kbc

    # Assuming llm is an LLM service instance
    cleaner = kbc.KBCGenerateCleanedTextSingle(llm=llm, lang='en')

    data = {'raw_chunk': 'Noisy text with errors...'}
    result = cleaner(data, input_key='raw_chunk')
    # Returns: {'raw_chunk': '...', '_cleaned_response': 'Cleaned text result'}
    ```
    """

    def __init__(self, llm=None, lang: str = 'en', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        # Initialize prompt template
        self.prompts = KnowledgeCleanerPrompt(lang=lang)

        # Initialize LLM serve with system prompt and formatter
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
        input_key: str = 'raw_chunk',
        **kwargs
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        raw_content = data.get(input_key, '')
        if not raw_content:
            return {**data, '_cleaned_response': raw_content}

        # Build prompt for the raw content
        user_prompt = self.prompts.build_prompt(raw_content)

        try:
            # Call LLM (system prompt and formatter already set in __init__)
            response = self._llm_serve(user_prompt)
            return {**data, '_cleaned_response': response}
        except Exception as e:
            LOG.warning(f'Failed to clean text: {e}')
            # Use raw content as fallback
            return {**data, '_cleaned_response': raw_content}


@data_register('data.kbc', rewrite_func='forward', _concurrency_mode='process')
def extract_cleaned_content_single(
    data: dict,
    output_key: str = 'cleaned_chunk',
) -> dict:
    """单条清洗内容提取函数。

该函数从单条LLM清洗响应中提取清洗后的文本内容，处理不同的响应格式。
支持从标签 <cleaned_start> 和 <cleaned_end> 之间提取内容，并清理中间字段。

Args:
    data (dict): 包含清洗响应的数据。
    output_key (str): 输出字段名，默认为 'cleaned_chunk'。

Returns:
    dict: 包含提取后清洗内容的数据，添加了 output_key 指定的字段。


Examples:
    ```python
    from lazyllm.tools.data.operators.knowledge_cleaning.kbc_text_cleaner import extract_cleaned_content_single

    data = {'_cleaned_response': '<cleaned_start>Clean text<cleaned_end>'}
    result = extract_cleaned_content_single(data, output_key='cleaned_chunk')
    # Returns: {'cleaned_chunk': 'Clean text'}
    ```
    """
    response = data.get('_cleaned_response', '')

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

    result = data.copy()
    result[output_key] = cleaned_text
    # Clean intermediate fields
    for key in ['_cleaned_response']:
        result.pop(key, None)
    return result
