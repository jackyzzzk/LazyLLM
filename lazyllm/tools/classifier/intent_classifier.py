from lazyllm.module import ModuleBase
from lazyllm.components import AlpacaPrompter
from lazyllm import pipeline, globals, switch
from lazyllm.tools.utils import chat_history_to_str
from typing import Dict, Union, Any, List, Optional
import json


en_prompt_classifier_template = '''
## role：Intent Classifier
You are an intent classification engine responsible for analyzing user input text based on dialogue information and determining a unique intent category.{user_prompt}

## Constrains:
You only need to reply with the name of the intent. Do not output any additional fields and do not translate it.{user_constrains}

## Attention:
{attention}

## Text Format
The input text is in JSON format, where 'human_input' contains the user's raw input and 'intent_list' contains a list of all intent names.

## Example
User: {{'human_input': 'What’s the weather like in Beijing tomorrow?', 'intent_list': ['Check Weather', 'Search Engine Query', 'View Surveillance', 'Report Summary', 'Chat']}}
Assistant: Check Weather
{user_examples}

## Conversation History
The chat history between the human and the assistant is stored within the <histories></histories> XML tags below.
<histories>
{history_info}
</histories>

Input text is as follows:
'''  # noqa E501

ch_prompt_classifier_template = '''
## role：意图分类器
你是一个意图分类引擎，负责根据对话信息分析用户输入文本并确定唯一的意图类别。{user_prompt}

## 限制:
你只需要回复意图的名字即可，不要额外输出其他字段，也不要进行翻译。{user_constrains}

## 注意:
{attention}

## 文本格式
输入文本为JSON格式，'human_input'中内容为用户的原始输入，'intent_list'为所有意图名列表

## 示例
User: {{'human_input': '北京明天天气怎么样？', 'intent_list': ['查看天气', '搜索引擎检索', '查看监控', '周报总结', '聊天']}}
Assistant:  查看天气
{user_examples}

## 历史对话
人类和助手之间的聊天记录存储在下面的 <histories></histories> XML 标记中。
<histories>
{history_info}
</histories>

输入文本如下:
'''  # noqa E501


class IntentClassifier(ModuleBase):
    """意图分类模块，用于根据输入文本在给定的意图列表中进行分类。
支持中英文自动选择提示模板，并可通过示例、提示、约束和注意事项增强分类效果。

Args:
    llm: 用于意图分类的大语言模型实例。
    intent_list (list): 可选，意图类别列表，例如 ["聊天", "天气", "问答"]。
    prompt (str): 可选，自定义提示语，插入到系统提示模板中。
    constrain (str): 可选，分类约束条件说明。
    attention (str): 可选，提示注意事项。
    examples (list[list[str, str]]): 可选，分类示例列表，每个元素为 [输入文本, 标签]。
    return_trace (bool): 是否返回执行过程的 trace，默认为 False。


Examples:
        >>> import lazyllm
        >>> from lazyllm.tools import IntentClassifier
        >>> classifier_llm = lazyllm.OnlineChatModule(source="openai")
        >>> chatflow_intent_list = ["Chat", "Financial Knowledge Q&A", "Employee Information Query", "Weather Query"]
        >>> classifier = IntentClassifier(classifier_llm, intent_list=chatflow_intent_list)
        >>> classifier.start()
        >>> print(classifier('What is the weather today'))
        Weather Query
        >>>
        >>> with IntentClassifier(classifier_llm) as ic:
        >>>     ic.case['Weather Query', lambda x: '38.5°C']
        >>>     ic.case['Chat', lambda x: 'permission denied']
        >>>     ic.case['Financial Knowledge Q&A', lambda x: 'Calling Financial RAG']
        >>>     ic.case['Employee Information Query', lambda x: 'Beijing']
        ...
        >>> ic.start()
        >>> print(ic('What is the weather today'))
        38.5°C
    """
    def __init__(self, llm, intent_list: list = None,
                 *, prompt: str = '', constrain: str = '', attention: str = '',
                 examples: Optional[list[list[str, str]]] = None, return_trace: bool = False) -> None:
        super().__init__(return_trace=return_trace)
        self._intent_list = intent_list or []
        self._llm = llm
        self._prompt, self._constrain, self._attention, self._examples = prompt, constrain, attention, examples or []
        if self._intent_list:
            self._init()

    def _init(self):
        def choose_prompt():
            # Use chinese prompt if intent elements have chinese character, otherwise use english version
            for ele in self._intent_list:
                for ch in ele:
                    # chinese unicode range
                    if '\u4e00' <= ch <= '\u9fff':
                        return ch_prompt_classifier_template
            return en_prompt_classifier_template

        example_template = '\nUser: {{{{"human_input": "{inp}", "intent_list": {intent}}}}}\nAssistant: {label}\n'
        examples = ''.join([example_template.format(
            inp=input, intent=self._intent_list, label=label) for input, label in self._examples])
        prompt = choose_prompt().replace(
            '{user_prompt}', f' {self._prompt}').replace('{attention}', self._attention).replace(
            '{user_constrains}', f' {self._constrain}').replace('{user_examples}', f' {examples}')
        self._llm = self._llm.share(prompt=AlpacaPrompter(dict(system=prompt, user='${input}')
                                                          ).pre_hook(self.intent_promt_hook)).used_by(self._module_id)
        self._impl = pipeline(self._llm, self.post_process_result)

    def intent_promt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: List[Union[List[str], Dict[str, Any]]] = [],  # noqa B006
        tools: Union[List[Dict[str, Any]], None] = None,
        label: Union[str, None] = None,
    ):
        """意图分类的预处理 Hook。
将输入文本与意图列表打包为 JSON，并生成历史对话信息字符串。

Args:
    input (str | List | Dict | None): 输入文本，仅支持字符串类型。
    history (List): 历史对话记录，默认为空列表。
    tools (List[Dict] | None): 工具信息，可选。
    label (str | None): 标签，可选。

**Returns:**

- tuple: 输入数据字典, 历史记录列表, 工具信息, 标签
"""
        input_json = {}
        if isinstance(input, str):
            input_json = {'human_input': input, 'intent_list': self._intent_list}
        else:
            raise ValueError(f'Unexpected type for input: {type(input)}')

        history_info = chat_history_to_str(history)
        history = []
        input_text = json.dumps(input_json, ensure_ascii=False)
        return dict(history_info=history_info, input=input_text), history, tools, label

    def post_process_result(self, input):
        """意图分类结果的后处理。
如果结果在意图列表中则直接返回，否则返回意图列表的第一个元素。

Args:
    input (str): 分类模型输出结果。

**Returns:**

- str: 最终的分类标签。
"""
        input = input.strip()
        return input if input in self._intent_list else self._intent_list[0]

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if llm_chat_history is not None and self._llm._module_id not in globals['chat_history']:
            globals['chat_history'][self._llm._module_id] = llm_chat_history
        return self._impl(input)

    def __enter__(self):
        assert not self._intent_list, 'Intent list is already set'
        self._sw = switch()
        self._sw.__enter__()
        return self

    @property
    def case(self):
        return switch.Case(self)

    @property
    def submodules(self):
        submodule = []
        if isinstance(self._impl, switch):
            self._impl.for_each(lambda x: isinstance(x, ModuleBase), lambda x: submodule.append(x))
        return super().submodules + submodule

    # used by switch.Case
    def _add_case(self, cond, func):
        assert isinstance(cond, str), 'intent must be string'
        self._intent_list.append(cond)
        self._sw.case[cond, func]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sw.__exit__(exc_type, exc_val, exc_tb)
        self._init()
        self._sw._set_conversion(self._impl)
        self._impl = self._sw
