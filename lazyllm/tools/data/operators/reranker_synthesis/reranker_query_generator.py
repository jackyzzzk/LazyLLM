import json
from typing import List, Optional

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.reranker_synthesis import RerankerQueryGeneratorPrompt

# Get or create reranker group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


def _clean_json_block(text: str) -> str:
    return text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()


class RerankerGenerateQueries(reranker):
    """基于给定文本生成多条检索查询（query）的算子。

该算子使用 RerankerQueryGeneratorPrompt 构造提示词，
调用 LLM 生成不同难度等级的查询语句。
生成结果通过 JsonFormatter 解析后，
以 JSON 字符串形式保存在 '_query_response' 字段中。

若输入 passage 为空或生成失败，则返回空响应字段。

Args:
    llm_serving: 语言模型服务实例
    lang (str): 查询生成语言，默认 'zh'
    num_queries (int): 生成查询数量，默认 3
    difficulty_levels (List[str]): 查询难度等级列表，默认 ['easy', 'medium', 'hard']
    **kwargs (dict): 其他可选参数，传递给父类。


Examples:
    ```python
    op = RerankerGenerateQueries(
        llm_serving=my_llm,
        lang='en',
        num_queries=5,
        difficulty_levels=['easy', 'hard']
    )

    result = op({'passage': 'Large language models are widely used in NLP.'})
    print(result['_query_response'])
    ```
    """
    def __init__(
        self,
        llm_serving=None,
        lang: str = 'zh',
        num_queries: int = 3,
        difficulty_levels: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.num_queries = num_queries
        self.difficulty_levels = difficulty_levels or ['easy', 'medium', 'hard']
        self.prompt_template = RerankerQueryGeneratorPrompt(lang=lang)

        # Initialize LLM serve with system prompt and formatter
        if llm_serving is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm_serving.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(
        self,
        data: dict,
        input_key: str = 'passage',
        **kwargs
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM serving is not configured')

        passage = data.get(input_key, '')
        if not passage:
            return {**data, '_query_response': ''}

        # Build user prompt from passage
        user_prompt = self.prompt_template.build_prompt(
            passage=passage,
            num_queries=self.num_queries,
            difficulty_levels=self.difficulty_levels
        )

        try:
            result = self._llm_serve(user_prompt)
            # JsonFormatter already parses JSON, handle both str and parsed result
            if isinstance(result, str):
                response = result
            else:
                response = json.dumps(result, ensure_ascii=False)
            return {**data, '_query_response': response}
        except Exception as e:
            LOG.warning(f'Failed to generate queries: {e}')
            return {**data, '_query_response': ''}


class RerankerParseQueries(reranker):
    """解析 LLM 生成的查询结果，并展开为多条训练样本数据。

该算子读取 '_query_response' 字段中的 JSON 内容，
解析得到查询列表（支持 list 或 {'queries': [...]} 结构）。
每条查询会生成一条新的数据记录，包含：

- query: 查询文本
- difficulty: 难度等级（默认 'medium'）
- pos: 正样本文本列表（原始 passage）

同时会清理中间字段 '_query_response' 等。

Args:
    input_key (str): 原始文本字段名，默认 'passage'
    output_query_key (str): 输出查询字段名，默认 'query'
    **kwargs (dict): 其他可选参数，传递给父类。


Examples:
    ```python
    op = RerankerParseQueries(input_key='passage', output_query_key='query')

    data = {
        'passage': 'Large language models are widely used in NLP.',
        '_query_response': '[{"query": "What are LLMs used for?", "difficulty": "easy"}]'
    }

    rows = op(data)
    for row in rows:
        print(row['query'], row['difficulty'], row['pos'])
    ```
    """
    def __init__(
        self,
        input_key: str = 'passage',
        output_query_key: str = 'query',
        **kwargs
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.input_key = input_key
        self.output_query_key = output_query_key

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> List[dict]:
        response = data.get('_query_response', '')
        if not response:
            return []

        passage = data.get(self.input_key, '')
        expanded_rows = []

        try:
            parsed = json.loads(_clean_json_block(response))
            queries = parsed if isinstance(parsed, list) else parsed.get('queries', [])

            for query_item in queries:
                if isinstance(query_item, dict):
                    query = query_item.get('query', '')
                    difficulty = query_item.get('difficulty', 'medium')
                else:
                    query = str(query_item)
                    difficulty = 'medium'

                if query.strip():
                    new_row = data.copy()
                    new_row[self.output_query_key] = query.strip()
                    new_row['difficulty'] = difficulty
                    new_row['pos'] = [passage]  # Positive sample is the source passage
                    # Clean up intermediate fields
                    new_row.pop('_query_prompt', None)
                    new_row.pop('_query_response', None)
                    expanded_rows.append(new_row)

        except Exception as e:
            LOG.warning(f'Failed to parse LLM response: {e}')
            return []

        return expanded_rows
