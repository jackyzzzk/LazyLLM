import copy
import re
from lazyllm.components.formatter import JsonFormatter
from lazyllm.thirdparty import rapidfuzz
from .eval_base import BaseEvaluator


class LLMContextRecall(BaseEvaluator):
    """用于评估回答中的每一句话是否可以归因于检索到的上下文的指标类。

该模块使用语言模型判断回答中的每个句子是否得到上下文的支持，通过二元值进行评分（1 表示支持，0 表示不支持或矛盾），最终计算平均回忆得分。

Args:
    llm (ModuleBase): 用于执行上下文一致性判断的语言模型。
    eval_prompt (str, 可选): 指导模型评估的自定义提示词。
    prompt_lang (str): 默认提示词语言，'en' 表示英文，'zh' 表示中文。
    retry (int): 评估失败时的最大重试次数。
    concurrency (int): 并发评估的任务数量。


Examples:
    >>> from lazyllm.components import LLMContextRecall
    >>> evaluator = LLMContextRecall(llm=YourLLM(), prompt_lang="en")
    >>> data = {
    ...     "question": "What is Photosynthesis?",
    ...     "answer": "Photosynthesis was discovered in the 1780s. It occurs in chloroplasts.",
    ...     "context_retrieved": [
    ...         "Photosynthesis occurs in chloroplasts.",
    ...         "Light reactions produce ATP using sunlight."
    ...     ]
    ... }
    >>> result = evaluator([data])
    >>> print(result)
    ... 0.5  # Final recall score averaged over statement evaluations
    """
    _default_eval_prompt_en = (
        '[Task Description]\n'
        'Given a context, and an answer, analyze each sentence in the answer and '
        'classify if the sentence can be attributed to the given context or not:\n'
        'Fully supported by context: 1\n'
        'Unsupported/contradictory: 0\n'
        '[Output Requirements]\n'
        '1. JSON format with array of objects\n'
        '2. Each object contains:\n'
        '    - "statement": Original text\n'
        '    - "reason": the reason why it is scored 1/0\n'
        '    - "score": 1 or 0\n'
        '3. Wrap output in ```json code block\n'
        '[Example Input]\n'
        'Question: What is Photosynthesis?'
        'Context: Photosynthesis occurs in chloroplasts. Light reactions produce ATP using sunlight.\n'
        'Statements: Photosynthesis was discovered in 1780s. It occurs in chloroplasts and produce ATP using sunlight.\n'
        '[Example Output]\n'
        '[{"statement": "Photosynthesis was discovered in 1780s", '
        '"reason": "The time when photosynthesis discovered was not mentioned in the given context","score": 0},'
        ' {"statement": "It occurs in chloroplasts and produce ATP using sunlight.", '
        '"reason": "The exact sentence is present in the given context", "score": 1}]\n'
    )
    _default_eval_prompt_zh = (
        '[任务描述]\n'
        '给定一个上下文和一个答案，分析答案中的每个句子并判断该句子是否可以归因于给定的上下文:\n'
        '完全受上下文支持:1\n'
        '不支持/矛盾:0\n'
        '[输出要求]\n'
        '1. 带有对象数组的 JSON 格式\n'
        '2. 每个对象包含:\n'
        ' - "statement":原始文本\n'
        ' - "reason":评分原因\n'
        ' - "score":1 或 0\n'
        '3. 将输出包裹在 ```json 代码块中\n'
        '[示例输入]\n'
        'question:什么是光合作用？'
        'context:光合作用发生在叶绿体中，利用阳光产生 ATP。\n'
        'statement:光合作用于 1780 年代被发现。光合作用发生在叶绿体中，并利用阳光产生 ATP。\n'
        '[示例输出]\n'
        '[{"statement": "光合作用于 1780 年代被发现", "reason": "给定上下文中未提及发现光合作用被发现的时间","score": 0},'
        ' {"statement": "光合作用发生在叶绿体中，并利用阳光产生 ATP。", "reason": "给定上下文中存在确切的句子", "score": 1}]\n'
    )

    def __init__(self, llm, eval_prompt=None, prompt_lang='en', retry=3, concurrency=1):
        super().__init__(concurrency, retry)
        if prompt_lang == 'zh':
            default_eval_prompt = eval_prompt or self._default_eval_prompt_zh
        else:
            default_eval_prompt = eval_prompt or self._default_eval_prompt_en
        self._llm = llm.prompt(default_eval_prompt).formatter(JsonFormatter()) if llm else None
        self._necessary_keys = ['question', 'answer', 'context_retrieved']

    def _validate_eval_result(self, result):
        return (
            isinstance(result, list)
            and len(result) > 0
            and all(isinstance(i, dict) and 'score' in i for i in result)
        )

    def _post_processor(self, eval_result):
        if isinstance(eval_result, dict):
            eval_result = [eval_result]
        return eval_result

    def _process_one_data_impl(self, data):
        res = copy.deepcopy(data)
        context = '\n'.join(data['context_retrieved'])

        query = f'question: {data["question"]}\ncontext: {context}\nstatement: {data["answer"]}'
        eval_result = self._execute_with_retries(
            query, self._llm, self._validate_eval_result, self._post_processor)
        scores = [result['score'] for result in eval_result]

        res['final_score'] = round(sum(scores) / len(scores), 4) if scores else 0.0
        return res

class NonLLMContextRecall(BaseEvaluator):
    """基于字符串模糊匹配的非LLM上下文回忆指标类。

该模块通过 Levenshtein 距离计算检索到的上下文与参考上下文的相似度，并给出回忆得分。可选择输出二值得分（是否存在足够相似的匹配）或平均匹配度得分。

Args:
    th (float): 相似度阈值（范围为0到1），值越高表示匹配越严格。
    binary (bool): 若为True，则只判断是否有任一匹配超过阈值；若为False，则输出所有匹配的平均得分。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。


Examples:
    >>> from lazyllm.components import NonLLMContextRecall
    >>> evaluator = NonLLMContextRecall(th=0.8, binary=True)
    >>> data = {
    ...     "context_retrieved": [
    ...         "Photosynthesis uses sunlight to produce sugar.",
    ...         "It takes place in chloroplasts."
    ...     ],
    ...     "context_reference": [
    ...         "Photosynthesis occurs in chloroplasts."
    ...     ]
    ... }
    >>> result = evaluator([data])
    >>> print(result)
    ... 1.0  # At least one retrieved context is similar enough
    """
    def __init__(self, th=0.5, binary=True, retry=3, concurrency=1):
        super().__init__(concurrency, retry)
        self._binary = binary
        self._threshold = th
        self._necessary_keys = ['context_retrieved', 'context_reference']

    def _calc_levenshtein_distance(self, reference, context):
        return 1 - rapidfuzz.distance.Levenshtein.normalized_distance(reference, context)

    def _calc_context_recall(self, data):
        contexts, reference = data['context'], data['reference']
        scores = []
        for context in contexts:
            score = self._calc_levenshtein_distance(reference, context)
            scores.append(score)
        return scores

    def _compute_scores(self, scores):
        binary_scores = [1 if score > self._threshold else 0 for score in scores]

        if self._binary:
            return 1.0 if sum(binary_scores) > 0 else 0.0
        if len(binary_scores) > 0:
            return sum(binary_scores) / len(binary_scores)
        return 0

    def _process_one_data_impl(self, data):
        res = copy.deepcopy(data)
        scores = []
        for reference in data['context_reference']:
            input_data = {'context': data['context_retrieved'], 'reference': reference}
            eval_result = self._execute_with_retries(input_data, self._calc_context_recall)
            scores.append(self._compute_scores(eval_result))

        res['final_score'] = round(sum(scores) / len(scores), 4) if scores else 0.0
        return res

class ContextRelevance(BaseEvaluator):
    """基于句子级匹配的非LLM上下文相关性评估器。

该模块将检索到的上下文与参考上下文分别按句子划分，并统计检索内容中与参考完全一致的句子数量，从而计算相关性得分。

Args:
    splitter (str): 句子分隔符，默认为中文句号 "。"，英文可设置为 "."。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。


Examples:
    >>> from lazyllm.components import ContextRelevance
    >>> evaluator = ContextRelevance(splitter='.')
    >>> data = {
    ...     "context_retrieved": [
    ...         "Photosynthesis occurs in chloroplasts. It produces glucose."
    ...     ],
    ...     "context_reference": [
    ...         "Photosynthesis occurs in chloroplasts. It requires sunlight. It produces glucose."
    ...     ]
    ... }
    >>> result = evaluator([data])
    >>> print(result)
    ... 0.6667  # 2 of 3 retrieved sentences match
    """
    def __init__(self, splitter='。', retry=3, concurrency=1):
        super().__init__(concurrency, retry)
        self._splitter = splitter
        self._necessary_keys = ['context_retrieved', 'context_reference']

    def _calc_context_relevance(self, data):
        sentences_retrieved, sentences_reference = data['context'], data['reference']
        scores = [0] * len(sentences_retrieved)
        for i, sentence in enumerate(sentences_retrieved):
            if sentence in sentences_reference:
                scores[i] = 1
        return scores

    def _paragraphs_to_sentences(self, paragraphs):
        sentences = []
        pattern = rf'{re.escape(self._splitter)}+'
        for paragraph in paragraphs:
            sentences.extend([s.strip() for s in re.split(pattern, paragraph) if s.strip()])
        return sentences

    def _process_one_data_impl(self, data):
        res = copy.deepcopy(data)
        retrieved = self._paragraphs_to_sentences(data['context_retrieved'])
        reference = self._paragraphs_to_sentences(data['context_reference'])

        input_data = {'context': retrieved, 'reference': reference}
        eval_result = self._execute_with_retries(input_data, self._calc_context_relevance)
        total_score = sum(eval_result)

        res['final_score'] = round(total_score / len(eval_result), 4) if eval_result else 0.0
        return res
