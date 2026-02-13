from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class ChatPrompter(LazyLLMPrompterBase):
    """用于多轮对话的大模型Prompt构造器，继承自 `LazyLLMPrompterBase`。

支持工具调用、历史对话与自定义指令模版。支持传入 system/user 拆分的指令结构，自动合并为统一模板。支持额外字段注入和打印提示信息。

Args:
    instruction (Option[str | Dict[str, str]]): Prompt模板指令，可为字符串或包含 `system` 和 `user` 的字典。若为字典，将自动拼接并注入特殊标记分隔符。
    extra_keys (Option[List[str]]): 额外的字段列表，用户输入中的内容会被插入对应槽位，用于丰富上下文。
    show (bool): 是否打印生成的Prompt，默认False。
    tools (Option[List]): 可选的工具列表，用于FunctionCall任务，默认None。
    history (Option[List[List[str]]]): 可选的历史对话，用于对话记忆，格式为[[user, assistant], ...]，默认None。


Examples:
    >>> from lazyllm import ChatPrompter

    - Simple instruction string
    >>> p = ChatPrompter('hello world')
    >>> p.generate_prompt('this is my input')
    'You are an AI-Agent developed by LazyLLM.hello world\\nthis is my input\\n'

    >>> p.generate_prompt('this is my input', return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world'}, {'role': 'user', 'content': 'this is my input'}]}

    - Using extra_keys
    >>> p = ChatPrompter('hello world {instruction}', extra_keys=['knowledge'])
    >>> p.generate_prompt({
    ...     'instruction': 'this is my ins',
    ...     'input': 'this is my inp',
    ...     'knowledge': 'LazyLLM-Knowledge'
    ... })
    'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\nthis is my inp\\n'

    - With conversation history
    >>> p.generate_prompt({
    ...     'instruction': 'this is my ins',
    ...     'input': 'this is my inp',
    ...     'knowledge': 'LazyLLM-Knowledge'
    ... }, history=[['s1', 'e1'], ['s2', 'e2']])
    'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\ns1|e1\\ns2|e2\\nthis is my inp\\n'

    - Using dict format for system/user instructions
    >>> p = ChatPrompter(dict(system="hello world", user="this is user instruction {input}"))
    >>> p.generate_prompt({'input': "my input", 'query': "this is user query"})
    'You are an AI-Agent developed by LazyLLM.hello world\\nthis is user instruction my input this is user query\\n'

    >>> p.generate_prompt({'input': "my input", 'query': "this is user query"}, return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input this is user query'}]}
    """
    def __init__(self, instruction: Union[None, str, Dict[str, str]] = None, extra_keys: Union[None, List[str]] = None,
                 show: bool = False, tools: Optional[List] = None, history: Optional[List[List[str]]] = None,
                 *, enable_system: bool = True):
        super(__class__, self).__init__(show, tools=tools, history=history, enable_system=enable_system)
        if isinstance(instruction, dict):
            splice_instruction = instruction.get('system', '') + \
                ChatPrompter.ISA + instruction.get('user', '') + ChatPrompter.ISE
            instruction = splice_instruction
        instruction_template = f'{instruction}\n{{extra_keys}}\n'.replace(
            '{extra_keys}', LazyLLMPrompterBase._get_extro_key_template(extra_keys)) if instruction else ''
        self._init_prompt('{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{user}{input}\n{eoh}{soa}\n',
                          instruction_template)

    @property
    def _split(self): return f'{self._soa}\n' if self._soa else None
