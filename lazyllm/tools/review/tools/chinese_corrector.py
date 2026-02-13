import re
import difflib
from typing import List, Optional, Dict, Any

import lazyllm

from lazyllm import AutoModel, warp, package
from ....module import LLMBase

DEFAULT_INSTRUCTION = '纠正输入句子中的语法错误，并只输出正确的句子，绝对不允许输出其他内容，' \
                      '例如：输入"我喜欢编程成"，输出"我喜欢编程"输入句子为：{sentence}'
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_TEMPERATURE = 0.6


def get_errors(corrected_text, origin_text):  # noqa: C901
    """比较修正文本和原始文本，找出其中的错误位置和内容。

使用序列匹配算法比较两个文本的差异，返回错误列表，每个错误包含原始字符、修正字符和位置信息。

Args:
    corrected_text (str): 修正后的文本。
    origin_text (str): 原始文本。

Returns:
    list: 错误列表，每个元素为 (orig_char, corr_char, pos) 的元组，其中：
        - orig_char (str): 原始字符，如果是插入错误则为空字符串。
        - corr_char (str): 修正字符，如果是删除错误则为空字符串。
        - pos (int): 错误在原始文本中的位置。


Examples:
        >>> from lazyllm.tools.review.tools.chinese_corrector import get_errors
        >>> errors = get_errors("我喜欢编程", "我喜欢编程成")
        >>> print(errors)
        [('', '成', 6)]
    """
    errors = []
    unk_tokens = set([' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', ''])

    def add_error(orig_char, corr_char, pos):
        if orig_char not in unk_tokens and corr_char not in unk_tokens:
            errors.append((orig_char, corr_char, pos))

    matcher = difflib.SequenceMatcher(None, origin_text, corrected_text)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        origin_part = origin_text[i1:i2]
        corrected_part = corrected_text[j1:j2]

        min_len = min(len(origin_part), len(corrected_part))

        for idx in range(min_len):
            add_error(origin_part[idx], corrected_part[idx], i1 + idx)

        for idx in range(min_len, len(origin_part)):
            add_error(origin_part[idx], '', i1 + idx)

        insert_pos = i1 + len(origin_part) if tag == 'replace' else i1
        for idx in range(min_len, len(corrected_part)):
            add_error('', corrected_part[idx], insert_pos)

    return sorted(errors, key=lambda x: x[2])


class ChineseCorrector:
    """中文文本纠错器，使用大语言模型对中文句子进行语法和拼写纠错。

通过配置不同的语言模型，可以对单个句子或批量句子进行纠错，并返回纠错结果和错误详情。

Args:
    llm: 可选，大语言模型实例。如果为None，则使用默认模型。
    base_url (str): 可选，模型服务的基础URL。
    model (str): 可选，使用的模型名称。
    api_key (str): 可选，API密钥，默认为'null'。
    source (str): 模型来源，默认为'openai'。


Examples:
        >>> import lazyllm
        >>> from lazyllm.tools.review.tools.chinese_corrector import ChineseCorrector
        >>> corrector = ChineseCorrector()
        >>> result = corrector.correct("我喜欢编程成")
        >>> print(result)
        {'source': '我喜欢编程成', 'target': '我喜欢编程', 'errors': [('成', '', 6)]}
        >>>
        >>> results = corrector.correct_batch(["句子1", "句子2"])
        >>> print(results)
        [{'source': '句子1', 'target': '修正后句子1', 'errors': [...]}, ...]
    """
    def __init__(self, llm: Optional[LLMBase] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, api_key: Optional[str] = 'null',
                 source: str = 'openai', **_: Any):
        if llm:
            base_llm = llm
        else:
            base_llm = AutoModel(source=source, model=model)
        self.base_llm = base_llm.prompt(lazyllm.AlpacaPrompter(DEFAULT_INSTRUCTION))

    def _predict(self, sentences: List[str], max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, **kwargs) -> List[Dict[str, Any]]:
        if not sentences:
            return []

        llm_kwargs = {
            'max_tokens': max_tokens or DEFAULT_MAX_TOKENS,
            'temperature': temperature if temperature is not None else DEFAULT_TEMPERATURE
        }

        llm_kwargs.update(kwargs)

        results: List[Dict[str, Any]] = []
        for sentence in sentences:
            try:
                response = self.base_llm(
                    dict(sentence=sentence),
                    stream_output=False,
                    **llm_kwargs,
                )
                response = self._post_process(response, sentence)
                errors = get_errors(response, sentence)
            except Exception as e:
                lazyllm.LOG.error(
                    f'Error predicting sentence {sentence[:50]}{"..." if len(sentence) > 50 else ""}. '
                    f'with max_tokens: {max_tokens}, temperature: {temperature}'
                    f'Error: {e}'
                )
                response = ''
                errors = []
            results.append(
                {
                    'source': sentence,
                    'target': response,
                    'errors': errors,
                }
            )

        return results

    def correct(self, sentence: str, **kwargs) -> Dict[str, Any]:
        """对单个中文句子进行语法和拼写纠错。

使用配置的语言模型对输入句子进行纠错，并返回包含原始文本、纠错文本和错误详情的结果字典。

Args:
    sentence (str): 需要纠错的中文句子。
    **kwargs: 其他传递给语言模型的参数，如max_tokens、temperature等。

Returns:
    dict: 包含以下键的字典：
        - source (str): 原始输入句子。
        - target (str): 纠错后的句子。
        - errors (list): 错误列表，每个元素为 (orig_char, corr_char, pos) 的元组。
"""
        results = self._predict([sentence], **kwargs)
        return results[0] if results else {'source': sentence, 'target': sentence, 'errors': []}

    def correct_batch(self, sentences: List[str], batch_size: int = DEFAULT_BATCH_SIZE,
                      concurrency: Optional[int] = 2, **kwargs) -> List[Dict[str, Any]]:
        """批量对中文句子进行语法和拼写纠错。

使用并行处理对多个句子进行纠错，提高处理效率。返回包含每个句子纠错结果的列表。

Args:
    sentences (list): 需要纠错的中文句子列表。
    batch_size (int): 可选，批处理大小，默认为4。
    concurrency (int): 可选，并发数，默认为2。
    **kwargs: 其他传递给语言模型的参数，如max_tokens、temperature等。

Returns:
    list: 每个元素为包含纠错结果的字典列表，每个字典包含：
        - source (str): 原始输入句子。
        - target (str): 纠错后的句子。
        - errors (list): 错误列表，每个元素为 (orig_char, corr_char, pos) 的元组。
"""
        if not sentences:
            return []

        def process_sentence(sent: str) -> Dict[str, Any]:
            try:
                res = self._predict([sent], **kwargs)
                return res[0] if res else {'source': sent, 'target': sent, 'errors': []}
            except Exception as e:
                lazyllm.LOG.error(f'Error processing sentence: {e}')
                return {'source': sent, 'target': sent, 'errors': []}

        try:
            results_package = warp(process_sentence, _concurrent=concurrency)(package(sentences))
            results = list(results_package)
            return results
        except Exception as e:
            lazyllm.LOG.error(f'Error in warp processing: {e}')
            return [{'source': sent, 'target': sent, 'errors': []} for sent in sentences]

    def _post_process(self, response: str, origin: str) -> str:
        response = response.strip()
        match = re.search(r'</think\s*>(.*)', response, re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            response = re.sub(r'^<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        sentence_endings = ['。', '！', '？', '；', '：', '，', '、', '.', ',', '?', '!', ':']
        origin_ending = origin[-1] if origin[-1] in sentence_endings else None
        response_ending = response[-1] if response[-1] in sentence_endings else None
        if origin_ending and not response_ending:
            response += origin_ending
        elif not origin_ending and response_ending:
            response = response[:-1]
        return response
