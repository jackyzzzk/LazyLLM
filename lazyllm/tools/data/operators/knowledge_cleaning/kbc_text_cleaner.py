"""KBC Text Cleaner operator"""
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


class KBCTextCleaner(kbc):
    """
    Knowledge cleaner for RAG to make content more accurate, reliable and readable.
    知识清洗算子：对原始知识内容进行标准化处理。
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
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KnowledgeCleanerPrompt(lang=lang)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "知识清洗算子：对原始知识内容进行标准化处理，包括HTML标签清理、特殊字符规范化、"
                "链接处理和结构优化，提升RAG知识库的质量。"
            )
        elif lang == "en":
            return (
                "Knowledge Cleaning Operator: Standardizes raw HTML/text content for RAG quality improvement."
            )
        else:
            return "Knowledge cleaning operator for RAG content standardization."

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

    def forward(
            self,
            data: dict,
            input_key: str = "raw_chunk",
            output_key: str = "cleaned_chunk",
    ) -> dict:
        """
        Clean raw text content for a single item.

        Args:
            data: Single dict item
            input_key: Key for input raw content
            output_key: Key for output cleaned content

        Returns:
            Dict with cleaned content added
        """
        assert isinstance(data, dict), "Input data must be a dict"

        raw_content = data.get(input_key, "")
        formatted_prompt = self.prompt_template.build_prompt(raw_content)
        cleaned = self._generate_from_llm([formatted_prompt], "")[0]

        # Extract content between <cleaned_start> and <cleaned_end>
        if '<cleaned_start>' in str(cleaned) and '<cleaned_end>' in str(cleaned):
            cleaned_text = str(cleaned).split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
        else:
            cleaned_text = str(cleaned).strip()

        result = data.copy()
        result[output_key] = cleaned_text
        return result
