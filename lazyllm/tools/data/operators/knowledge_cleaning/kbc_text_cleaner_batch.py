"""KBC Text Cleaner Batch operator"""
import json
from typing import List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register
from ...prompts.kbcleaning import KnowledgeCleanerPrompt

# 复用已存在的 kbc 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCTextCleanerBatch(kbc):
    """
    Batch knowledge cleaner for RAG.
    批量知识清洗算子。
    """

    def __init__(
            self,
            llm=None,
            lang: str = "en",
            prompt_template=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.lang = lang
        self.prompts = KnowledgeCleanerPrompt(lang=lang)
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KnowledgeCleanerPrompt(lang=lang)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "批量知识清洗算子：对原始知识内容进行标准化处理。"
            )
        elif lang == "en":
            return (
                "Batch Knowledge Cleaning Operator for RAG content standardization."
            )
        else:
            return "Batch knowledge cleaning operator for RAG content standardization"

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm is None:
            raise ValueError("LLM is not configured")
        llm_serve = self.llm.share(prompt=system_prompt)
        llm_serve.start()
        results = []
        for prompt in user_prompts:
            results.append(llm_serve(prompt))
        return results

    def _reformat_prompt_from_path(self, chunk_path: str):
        """Load and reformat prompts from file path"""
        if chunk_path.endswith(".json"):
            with open(chunk_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
        elif chunk_path.endswith(".jsonl"):
            with open(chunk_path, 'r', encoding='utf-8') as f:
                file_data = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file format. Only .json and .jsonl are supported.")

        if not file_data or "raw_chunk" not in file_data[0]:
            raise KeyError("'raw_chunk' field not found in the input file.")

        raw_contents = [item.get("raw_chunk", "") for item in file_data]
        inputs = [self.prompts.build_prompt(content) for content in raw_contents]
        return raw_contents, inputs

    def forward(
            self,
            data: dict,
            input_key: str = "chunk_path",
            output_key: str = "cleaned_chunk_path",
    ) -> dict:
        """
        Clean text content from a single chunk file.

        Args:
            data: Single dict item
            input_key: Key for input chunk file path
            output_key: Key for output cleaned chunk file path

        Returns:
            Dict with cleaned chunk path added
        """
        assert isinstance(data, dict), "Input data must be a dict"

        chunk_path = data.get(input_key, "")

        if chunk_path:
            raw_chunks, formatted_prompts = self._reformat_prompt_from_path(chunk_path)
            cleaned = self._generate_from_llm(formatted_prompts, "")

            # Extract content between <cleaned_start> and <cleaned_end>
            cleaned_extracted = [
                text.split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
                if '<cleaned_start>' in str(text) and '<cleaned_end>' in str(text)
                else str(text).strip()
                for text in cleaned
            ]

            json_items = [{
                "raw_chunk": raw_chunk,
                "cleaned_chunk": cleaned_chunk
            } for raw_chunk, cleaned_chunk in zip(raw_chunks, cleaned_extracted)]

            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump(json_items, f, ensure_ascii=False, indent=4)
            LOG.info(f"Successfully cleaned contents in {chunk_path}")

        result = data.copy()
        result[output_key] = chunk_path
        return result
