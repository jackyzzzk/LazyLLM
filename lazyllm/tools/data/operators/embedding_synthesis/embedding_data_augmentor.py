"""
Embedding Data Augmentor Operators

This module provides operators for augmenting embedding training data through various techniques.
"""
import json
import random
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.embedding_synthesis import EmbeddingQueryAugmentPrompt

# Get or create embedding group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


def _clean_json_block(text: str) -> str:
    """Clean JSON code block markers from LLM output."""
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class EmbeddingQueryRewrite(embedding):
    """
    Operator for augmenting queries by rewriting with LLM.
    Generates semantically equivalent query variations.
    """

    def __init__(
        self,
        llm=None,
        num_augments: int = 2,
        lang: str = "zh",
        **kwargs
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.num_augments = num_augments
        self.lang = lang
        self.prompt_template = EmbeddingQueryAugmentPrompt(lang=lang)
        
        # Initialize LLM serve with system prompt and formatter
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> List[dict]:
        """
        Rewrite a single query and return list of augmented samples.
        
        Returns:
            List of dict with augmented queries (returns list to expand data)
        """
        if self._llm_serve is None:
            raise ValueError("LLM is not configured")

        query = data.get('query', '')
        if not query:
            return []

        user_prompt = self.prompt_template.build_prompt(query=query, num_rewrites=self.num_augments)

        try:
            result = self._llm_serve(user_prompt)
            
            # Parse result from JsonFormatter
            # Note: JsonFormatter has already parsed the response
            # If result is a string, it means parsing failed
            rewrites = []
            if isinstance(result, list):
                rewrites = result
            elif isinstance(result, dict):
                rewrites = result.get("rewrites", [])
            elif isinstance(result, str):
                # JsonFormatter failed to parse, skip
                LOG.warning(f"JsonFormatter failed to parse response, skipping: {result[:100]}...")
                return []

            # Create augmented samples
            augmented = []
            for rewrite in rewrites:
                rewrite_str = str(rewrite).strip()
                if rewrite_str and rewrite_str != query:
                    new_row = data.copy()
                    new_row['query'] = rewrite_str
                    new_row["is_augmented"] = True
                    new_row["augment_method"] = "query_rewrite"
                    augmented.append(new_row)
            
            return augmented
        except Exception as e:
            LOG.warning(f"Failed to rewrite query: {e}")
            return []


class EmbeddingSynonymReplace(embedding):
    """
    Operator for augmenting queries by random synonym replacement (rule-based).
    Simple character-level augmentation for Chinese, word-level for English.
    """

    def __init__(
        self,
        num_augments: int = 2,
        **kwargs
    ):
        # Rule-based operation, use process mode for CPU-bound tasks
        super().__init__(_concurrency_mode='process', **kwargs)
        self.num_augments = num_augments

    def forward(self, data: dict) -> List[dict]:
        """
        Apply synonym replacement augmentation to a single query.
        
        Returns:
            List of dict with augmented queries
        """
        query = data.get('query', '')
        if not query:
            return []

        augmented = []
        words = query.split()
        
        for _ in range(self.num_augments):
            if len(words) > 2:
                # Randomly swap two adjacent words as simple augmentation
                idx = random.randint(0, len(words) - 2)
                new_words = words.copy()
                new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
                new_query = " ".join(new_words)
                
                if new_query != query:
                    new_row = data.copy()
                    new_row['query'] = new_query
                    new_row["is_augmented"] = True
                    new_row["augment_method"] = "synonym_replace"
                    augmented.append(new_row)
            else:
                # Keep original if too short
                break

        return augmented


# Keep the original class for backward compatibility
class EmbeddingDataAugmentor(embedding):
    """
    Legacy operator for augmenting embedding training data.
    Uses the new pipeline operators internally.
    """

    def __init__(
        self,
        llm=None,
        augment_methods: Optional[List[str]] = None,
        num_augments: int = 2,
        lang: str = "zh",
        **kwargs
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.llm = llm
        self.augment_methods = augment_methods or ["query_rewrite"]
        self.num_augments = num_augments
        self.lang = lang
        LOG.info(f"Initializing {self.__class__.__name__} with methods: {self.augment_methods}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingDataAugmentor 算子用于增强 Embedding 训练数据。\n\n"
                "核心功能：\n"
                "- 查询改写：使用 LLM 生成语义等价的不同表达\n"
                "- 同义词替换：替换查询中的词汇为同义词\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- output_query_key: 增强后查询字段名（默认：'query'）\n\n"
                "输出：包含原始和增强样本的数据列表"
            )
        else:
            return (
                "EmbeddingDataAugmentor augments embedding training data.\n\n"
                "Features:\n"
                "- Query rewriting: Generate semantically equivalent expressions\n"
                "- Synonym replacement: Replace words with synonyms\n\n"
                "Input:\n"
                "- input_query_key: Query field name (default: 'query')\n"
                "- output_query_key: Augmented query field name (default: 'query')"
            )

    def forward_batch_input(
        self,
        inputs: List[dict],
        input_query_key: str = "query",
        output_query_key: str = "query",
        keep_original: bool = True,
        **kwargs
    ) -> List[dict]:
        """
        Augment the training data using pipeline operators.
        """
        from lazyllm import pipeline

        assert isinstance(inputs, list), "inputs must be a list of dict"

        LOG.info(f"Augmenting {len(inputs)} samples with methods: {self.augment_methods}")

        # Normalize input data
        normalized_inputs = []
        for item in inputs:
            normalized_item = item.copy()
            if input_query_key != 'query':
                normalized_item['query'] = item.get(input_query_key, '')
            normalized_inputs.append(normalized_item)

        results = []
        if keep_original:
            results.extend(inputs)

        # Apply each augmentation method using pipeline
        for method in self.augment_methods:
            LOG.info(f"Applying augmentation method: {method}")
            
            with pipeline() as ppl:
                if method == "query_rewrite":
                    ppl.augment = EmbeddingQueryRewrite(
                        llm=self.llm,
                        num_augments=self.num_augments,
                        lang=self.lang
                    )
                elif method == "synonym_replace":
                    ppl.augment = EmbeddingSynonymReplace(
                        num_augments=self.num_augments
                    )
                else:
                    LOG.warning(f"Unknown augmentation method: {method}, skipping...")
                    continue

            augmented = ppl(normalized_inputs)
            
            # Restore original key name if needed
            if output_query_key != 'query':
                for item in augmented:
                    item[output_query_key] = item.pop('query', '')
            
            results.extend(augmented)

        original_count = len(inputs) if keep_original else 0
        augmented_count = len(results) - original_count
        LOG.info(f"Augmentation completed: {original_count} original + {augmented_count} augmented = {len(results)} total")
        return results
