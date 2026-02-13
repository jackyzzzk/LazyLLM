import re
import json
import collections
from lazyllm import LOG

templates = dict(
    # Template used by Alpaca-LoRA.
    alpaca={
        'prompt': 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n',  # noqa E501
        'response_split': '### Response:',
    },
    # Template used by internLM
    puyu={
        'prompt': '<bos><|System|>:You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<eosys>\n<|Human|>:{instruction}<eoh>\n<|Assistant|>:please tell me what to do。ി\n<|Human|>:{input}<eoh>\n<|Assistant|>:', # noqa E501
        'response_split': '<|Assistant|>:',
    }
)

class Prompter(object):
    """用于生成模型输入的Prompt类，支持模板、历史对话拼接与响应抽取。

该类支持从字典、模板名称或文件中加载prompt配置，支持历史对话结构拼接（用于Chat类任务），
可灵活处理有/无history结构的prompt输入，适配非字典类型输入。

Args:
    prompt (Optional[str]): 模板Prompt字符串，支持格式化字段。
    response_split (Optional[str]): 对模型响应进行切分的分隔符，仅用于抽取模型回答。
    chat_prompt (Optional[str]): 多轮对话使用的Prompt模板，必须包含history字段。
    history_symbol (str): 表示历史对话字段的名称，默认为'llm_chat_history'。
    eoa (Optional[str]): 对话中 assistant/user 分隔符。
    eoh (Optional[str]): 多轮history中 user-assistant 分隔符。
    show (bool): 是否打印最终生成的Prompt，默认False。


Examples:
    >>> from lazyllm import Prompter

    >>> p = Prompter(prompt="Answer the following: {question}")
    >>> p.generate_prompt("What is AI?")
    'Answer the following: What is AI?'

    >>> p.generate_prompt({"question": "Define machine learning"})
    'Answer the following: Define machine learning'

    >>> p = Prompter(
    ...     prompt="Instruction: {instruction}",
    ...     chat_prompt="Instruction: {instruction}\\nHistory:\\n{llm_chat_history}",
    ...     history_symbol="llm_chat_history",
    ...     eoa="</s>",
    ...     eoh="|"
    ... )
    >>> p.generate_prompt(
    ...     input={"instruction": "Translate this."},
    ...     history=[["hello", "你好"], ["how are you", "你好吗"]]
    ... )
    'Instruction: Translate this.\\nHistory:\\nhello|你好</s>how are you|你好吗'

    >>> prompt_conf = {
    ...     "prompt": "Task: {task}",
    ...     "response_split": "---"
    ... }
    >>> p = Prompter.from_dict(prompt_conf)
    >>> p.generate_prompt("Summarize this article.")
    'Task: Summarize this article.'

    >>> full_output = "Task: Summarize this article.---This is the summary."
    >>> p.get_response(full_output)
    'This is the summary.'
    """
    def __init__(self, prompt=None, response_split=None, *, chat_prompt=None,
                 history_symbol='llm_chat_history', eoa=None, eoh=None, show=False):
        self._prompt, self._response_split = prompt, response_split
        self._chat_prompt = chat_prompt
        self._history_symbol, self._eoa, self._eoh = history_symbol, eoa, eoh
        self._show = show
        self._prompt_keys = list(set(re.findall(r'\{(\w+)\}', self._prompt))) if prompt else []
        if chat_prompt is not None:
            chat_keys = set(re.findall(r'\{(\w+)\}', self._chat_prompt))
            assert set(self._prompt_keys).issubset(chat_keys)
            assert chat_keys - set(self._prompt_keys) == set([self._history_symbol])
            self.use_history = True
        else:
            self.use_history = history_symbol in self._prompt_keys
            if self.use_history:
                self._prompt_keys.pop(self._prompt_keys.index(history_symbol))
                self._chat_prompt = self._prompt

    @classmethod
    def from_dict(cls, prompt, *, show=False):
        """通过字典配置初始化一个 Prompter 实例。

Args:
    prompt (Dict): 包含 prompt 相关字段的配置字典，需包含 `prompt` 键，其他为可选。
    show (bool): 是否显示生成的 prompt，默认为 False。

**Returns:**

- Prompter: 返回一个初始化的 Prompter 实例。
"""
        assert isinstance(prompt, dict)
        return cls(**prompt, show=show)

    @classmethod
    def from_template(cls, template_name, *, show=False):
        """根据模板名称加载 prompt 配置并初始化 Prompter 实例。

Args:
    template_name (str): 模板名称，必须在 `templates` 中存在。
    show (bool): 是否显示生成的 prompt，默认为 False。

**Returns:**

- Prompter: 返回一个初始化的 Prompter 实例。
"""
        return cls.from_dict(templates[template_name], show=show)

    @classmethod
    def from_file(cls, fname, *, show=False):
        """从 JSON 文件中读取配置并初始化 Prompter 实例。

Args:
    fname (str): JSON 配置文件路径。
    show (bool): 是否显示生成的 prompt，默认为 False。

Returns:
    Prompter: 返回一个初始化的 Prompter 实例。
"""
        with open(fname) as fp:
            return cls.from_dict(json.load(fp), show=show)

    @classmethod
    def empty(cls):
        """创建一个空的 Prompter 实例。

Returns:
    Prompter: 返回一个无 prompt 配置的 Prompter 实例。
"""
        return cls()

    def _is_empty(self):
        return self._prompt is None

    def generate_prompt(self, input, history=None, tools=None, label=None, show=False):
        """根据输入和可选的历史记录生成最终 Prompt。

Args:
    input (Union[str, Dict]): 用户输入。可以是字符串或包含多字段的字典。
    history (Optional[List[List[str]]]): 多轮对话历史，例如 [['u1', 'a1'], ['u2', 'a2']]。
    tools (Optional[Any]): 目前未支持工具调用，此字段必须为 None。
    label (Optional[str]): 附加在 prompt 末尾的标签，通常用于训练。
    show (bool): 是否显示生成的 prompt，默认 False。

Returns:
    str: 格式化后的 prompt 字符串。
"""
        if not self._is_empty():
            assert tools is None
            # datasets.formatting.formatting.LazyDict is used in transformers
            if not isinstance(input, collections.abc.Mapping):
                assert len(self._prompt_keys) == 1, (
                    f'invalid prompt `{self._prompt}` for <{type(input)}> input `{input}`')
                input = {self._prompt_keys[0]: input}
            try:
                if self.use_history and isinstance(history, list) and len(history) > 0:
                    assert isinstance(history[0], list), 'history must be list of list'
                    input[self._history_symbol] = self._eoa.join([self._eoh.join(h) for h in history])
                    input = self._chat_prompt.format(**input)
                else:
                    if self.use_history: input[self._history_symbol] = ''
                    input = self._prompt.format(**input)
            except Exception:
                raise RuntimeError(f'Generate prompt failed, and prompt is {self._prompt}; chat-prompt'
                                   f' is {self._chat_prompt}; input is {input}; history is {history}')
            if label: input += label
        if self._show or show: LOG.info(input)
        return input

    def get_response(self, response, input=None):
        """从 LLM 返回结果中提取模型的回答内容。

Args:
    response (str): 模型完整响应文本。
    input (Optional[str]): 如果模型输出以输入开头，将会自动去除输入部分。

Returns:
    str: 提取后的模型响应内容。
"""
        if input and response.startswith(input):
            return response[len(input):]
        return response if self._response_split is None else response.split(self._response_split)[-1]
