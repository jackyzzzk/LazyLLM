"""
Reranker Query Generator Operator

This operator generates high-quality queries from document passages for reranker model training.
该算子从文档段落生成高质量的查询，用于 Reranker 模型训练。
"""
import json
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register
from ...prompts.reranker_synthesis import RerankerQueryGeneratorPrompt

# 获取或创建 reranker 分组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


class RerankerQueryGenerator(reranker):
    """
    Generate high-quality queries from document passages for reranker training.
    从文档段落生成高质量查询用于 Reranker 训练。

    This operator uses LLM to analyze document content and generate queries
    of varying difficulty levels. Each passage can generate multiple queries,
    resulting in expanded training data (one-to-many).

    Args:
        llm_serving: LLM service for query generation
        num_queries: Number of queries to generate per passage (default: 3)
        lang: Language for prompts ("zh" or "en")
        difficulty_levels: List of difficulty levels (default: ["easy", "medium", "hard"])
    """

    def __init__(
            self,
            llm_serving=None,
            num_queries: int = 3,
            lang: str = "zh",
            difficulty_levels: Optional[List[str]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_serving = llm_serving
        self.num_queries = num_queries
        self.lang = lang
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        self._prompt_template = RerankerQueryGeneratorPrompt(lang=self.lang)
        LOG.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RerankerQueryGenerator 算子用于从文档段落生成高质量查询。\n\n"
                "核心功能：\n"
                "- 使用 LLM 分析文档内容并生成不同难度的查询\n"
                "- 支持多难度级别查询生成（简单、中等、困难）\n"
                "- 自动构建 query-passage 训练对\n\n"
                "输入参数：\n"
                "- input_key: 输入文档段落字段名（默认：'passage'）\n"
                "- output_query_key: 输出查询字段名（默认：'query'）\n\n"
                "输出：包含生成查询的数据列表"
            )
        else:
            return (
                "RerankerQueryGenerator generates high-quality queries from document passages.\n\n"
                "Features:\n"
                "- Uses LLM to analyze document content and generate queries of varying difficulty\n"
                "- Supports multiple difficulty levels (easy, medium, hard)\n"
                "- Automatically builds query-passage training pairs"
            )

    def _clean_json_block(self, text: str) -> str:
        """Clean JSON code block markers from LLM output."""
        return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def _generate_from_llm(self, user_prompts: List[str], system_prompt: str = "") -> List[str]:
        """Call LLM serving to generate responses."""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")

        llm = self.llm_serving.share(prompt=system_prompt)
        llm.start()
        results = []
        for prompt in user_prompts:
            try:
                response = llm(prompt)
                results.append(response)
            except Exception as e:
                LOG.warning(f"LLM call failed: {e}")
                results.append("")
        return results

    def forward(self, data, input_key: str = "passage", output_query_key: str = "query"):
        """
        Generate queries for a single passage.

        Args:
            data: Dict containing passage
            input_key: Key for input passage field
            output_query_key: Key for output query field

        Returns:
            List of dict with generated queries (one-to-many expansion)
        """
        assert isinstance(data, dict), "Input data must be a dict"

        passage = data.get(input_key, "")
        if not passage:
            LOG.warning(f"Empty passage in data, skipping")
            return []

        # Build prompt
        system_prompt = self._prompt_template.build_system_prompt()
        user_prompt = self._prompt_template.build_prompt(
            passage=passage,
            num_queries=self.num_queries,
            difficulty_levels=self.difficulty_levels
        )

        # Generate queries using LLM
        responses = self._generate_from_llm([user_prompt], system_prompt)
        response = responses[0] if responses else ""

        # Parse response and expand rows
        expanded_rows = []
        try:
            parsed = json.loads(self._clean_json_block(response))
            queries = parsed if isinstance(parsed, list) else parsed.get("queries", [])

            for query_item in queries:
                if isinstance(query_item, dict):
                    query = query_item.get("query", "")
                    difficulty = query_item.get("difficulty", "medium")
                else:
                    query = str(query_item)
                    difficulty = "medium"

                if query.strip():
                    new_row = data.copy()
                    new_row[output_query_key] = query.strip()
                    new_row["difficulty"] = difficulty
                    new_row["pos"] = [passage]  # Positive sample is the source passage
                    expanded_rows.append(new_row)

        except Exception as e:
            LOG.warning(f"Failed to parse LLM response: {e}")

        return expanded_rows
