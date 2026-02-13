import json
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.embedding_synthesis import EmbeddingQueryGeneratorPrompt


# Get or create embedding group
if (
    'data' in LazyLLMRegisterMetaClass.all_clses
    and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']
):
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


def _clean_json_block(item: str) -> str:
    return (
        item.strip()
        .removeprefix('```json')
        .removeprefix('```')
        .removesuffix('```')
        .strip()
    )

class EmbeddingGenerateQueries(embedding):
    """使用 LLM 生成查询的算子。

该算子调用语言模型服务，基于构建的提示生成查询。返回 JSON 格式的查询响应。

Args:
    llm: LLM 服务实例，用于生成查询。
    num_queries (int): 要生成的查询数量，默认为 3。
    lang (str): 语言，'zh' 表示中文，'en' 表示英文，默认为 'zh'。
    query_types (List[str], optional): 查询类型列表，默认为 ['factual', 'semantic', 'inferential']。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 输入数据，添加了 '_query_response' 字段包含生成的查询响应。


Examples:
    ```python
    from lazyllm.tools.data import embedding

    # Assuming llm is an LLM service instance
    generator = embedding.EmbeddingGenerateQueries(llm=llm, lang='zh')
    data = {'_query_prompt': 'Generate queries for: machine learning tutorial'}
    result = generator(data)
    # Returns data with '_query_response' field containing JSON queries
    ```
    """
    def __init__(
        self,
        llm=None,
        num_queries: int = 3,
        lang: str = 'zh',
        query_types: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompt_template = EmbeddingQueryGeneratorPrompt(lang=lang)
        self.num_queries = num_queries
        self.query_types = query_types or ['factual', 'semantic', 'inferential']
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

    def forward(
        self,
        data: dict,
        input_key: str = 'passage',
        **kwargs,
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        passage = data.get(input_key, '')
        if not passage:
            return {**data, '_query_response': ''}

        user_prompt = self.prompt_template.build_prompt(
            passage=passage,
            num_queries=self.num_queries,
            query_types=self.query_types,
        )
        if not user_prompt:
            return {**data, '_query_response': ''}

        try:
            result = self._llm_serve(user_prompt)

            if isinstance(result, str):
                response = result
            else:
                response = json.dumps(result, ensure_ascii=False)

            return {**data, '_query_response': response}

        except Exception as e:
            LOG.warning(f'Failed to generate queries: {e}')
            return {**data, '_query_response': ''}


class EmbeddingParseQueries(embedding):
    """解析生成的查询的算子。

该算子解析 LLM 生成的查询响应，将每条查询展开为独立的数据记录。

Args:
    input_key (str): 输入字段名，默认为 'passage'。
    output_query_key (str): 输出查询字段名，默认为 'query'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 解析后的查询列表，每个查询为一个独立的数据记录。


Examples:
    ```python
    from lazyllm.tools.data import embedding

    parser = embedding.EmbeddingParseQueries(input_key='passage', output_query_key='query')
    data = {'_query_response': '[{"query": "what is ML?", "type": "factual"}]', 'passage': 'Machine learning is...'}
    result = parser(data)
    # Returns list of expanded query records with 'query' and 'pos' fields
    ```
    """
    def __init__(
        self,
        input_key: str = 'passage',
        output_query_key: str = 'query',
        **kwargs,
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.input_key = input_key
        self.output_query_key = output_query_key

    def forward(
        self,
        data: dict,
        **kwargs,
    ) -> List[dict]:
        response = data.get('_query_response', '')
        if not response:
            return []

        passage = data.get(self.input_key, '')
        expanded_rows = []

        try:
            parsed = json.loads(_clean_json_block(response))
            queries = (
                parsed if isinstance(parsed, list)
                else parsed.get('queries', [])
            )

            for query_item in queries:
                if isinstance(query_item, dict):
                    query = query_item.get('query', '')
                    query_type = query_item.get('type', 'unknown')
                else:
                    query = str(query_item)
                    query_type = 'unknown'

                if query.strip():
                    new_row = data.copy()
                    new_row[self.output_query_key] = query.strip()
                    new_row['query_type'] = query_type
                    new_row['pos'] = [passage]

                    new_row.pop('_query_prompt', None)
                    new_row.pop('_query_response', None)

                    expanded_rows.append(new_row)

        except Exception as e:
            LOG.warning(f'Failed to parse query response: {e}')
            return []

        return expanded_rows
