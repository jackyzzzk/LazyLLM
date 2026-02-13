from typing import Dict, Union, Any, List, Callable, Optional
from ...common import LazyLLMRegisterMetaClass
from lazyllm import LOG
import json5 as json
from functools import reduce
import copy
import re

FC_PROMPT = '''{tool_start_token}tool name (one of {tool_names})
{tool_args_token}the input to the tool, in a JSON format representing the kwargs. Can only return json.
{tool_end_token}end of tool.
(e.g. {tool_start_token}tool_name\n{tool_args_token}{{'input': 'hello world', 'num_beams': 5}}\n{tool_end_token}).
'''

FC_PROMPT_PLACEHOLDER = '<!lazyllm-fc-prompt!>'


class LazyLLMPrompterBase(metaclass=LazyLLMRegisterMetaClass):
    """LazyLLM提示词基类，用于管理和生成模型提示词。

Args:
    show (bool): 是否显示生成的提示词，默认为False。
    tools (Optional[List]): 可用工具列表，默认为None。
    history (Optional[List]): 对话历史记录，默认为None。

Attributes:
    ISA (str): 指令分隔符起始标记 "<!lazyllm-spliter!>"。

    ISE (str): 指令分隔符结束标记 "</!lazyllm-spliter!>"。


Configuration Items:
    - system: 系统角色设定

    - sos/eos: 会话开始/结束标记

    - soh/eoh: 人类输入开始/结束标记

    - soa/eoa: AI回复开始/结束标记

    - soe/eoe: 工具执行结果开始/结束标记

    - tool_start_token/tool_end_token: 工具调用开始/结束标记

    - tool_args_token: 工具参数标记

"""
    ISA = '<!lazyllm-spliter!>'
    ISE = '</!lazyllm-spliter!>'

    def __init__(self, show=False, tools=None, history=None, *, enable_system: bool = True):
        self._set_model_configs(system='You are an AI-Agent developed by LazyLLM.', sos='',
                                soh='', soa='', eos='', eoh='', eoa='')
        self._show = show
        self._tools = tools
        self._pre_hook = None
        self._history = history or []
        self._enable_system = enable_system

    def _init_prompt(self, template: str, instruction_template: str, split: Union[None, str] = None):
        self._template = template
        self._instruction_template = instruction_template
        if split:
            assert not hasattr(self, '_split')
            self._split = split

    @staticmethod
    def _get_extro_key_template(extra_keys, prefix='Here are some extra messages you can referred to:\n\n'):
        if extra_keys:
            if isinstance(extra_keys, str): extra_keys = [extra_keys]
            assert isinstance(extra_keys, (tuple, list)), 'Only str, tuple[str], list[str] are supported'
            return prefix + ''.join([f'### {k}:\n{{{k}}}\n\n' for k in extra_keys])
        return ''

    def _handle_tool_call_instruction(self, instruction, tools):
        tool_dict = {}
        for key in ['tool_start_token', 'tool_args_token', 'tool_end_token']:
            if getattr(self, f'_{key}', None) and key in instruction:
                tool_dict[key] = getattr(self, f'_{key}')
        if 'tool_names' in instruction: tool_dict['tool_names'] = self._get_tools_name(tools)
        return reduce(lambda s, kv: s.replace(f'{{{kv[0]}}}', kv[1]), tool_dict.items(), instruction)

    def _set_model_configs(self, system: str = None, sos: Union[None, str] = None, soh: Union[None, str] = None,
                           soa: Union[None, str] = None, eos: Union[None, str] = None,
                           eoh: Union[None, str] = None, eoa: Union[None, str] = None,
                           soe: Union[None, str] = None, eoe: Union[None, str] = None,
                           separator: Union[None, str] = None, plugin: Union[None, str] = None,
                           interpreter: Union[None, str] = None, stop_words: Union[None, List[str]] = None,
                           tool_start_token: Union[None, str] = None, tool_end_token: Union[None, str] = None,
                           tool_args_token: Union[None, str] = None):

        local = locals()
        for name in ['system', 'sos', 'soh', 'soa', 'eos', 'eoh', 'eoa', 'soe', 'eoe', 'tool_start_token',
                     'tool_end_token', 'tool_args_token']:
            if local[name] is not None: setattr(self, f'_{name}', local[name])

    def _get_tools(self, tools, *, return_dict):
        return tools if return_dict else '### Function-call Tools. \n\n' +\
            f'{json.dumps(tools, ensure_ascii=False)}\n\n' if tools else ''

    def _get_tools_name(self, tools):
        return json.dumps([t['function']['name'] for t in tools], ensure_ascii=False) if tools else ''

    def _get_histories(self, history, *, return_dict):  # noqa: C901
        if not self._history and not history: return ''
        if return_dict:
            content = []
            for item in self._history + (history or []):
                if isinstance(item, list):
                    assert len(item) <= 2, 'history item length cannot be greater than 2'
                    if len(item) > 0: content.append({'role': 'user', 'content': item[0]})
                    if len(item) > 1: content.append({'role': 'assistant', 'content': item[1]})
                elif isinstance(item, dict):
                    content.append(item)
                else:
                    LOG.error(f'history: {history}')
                    raise ValueError('history must be a list of list or dict')
            return content
        else:
            ret = ''.join([f'{self._soh}{h}{self._eoh}{self._soa}{a}{self._eoa}' for h, a in self._history])
            if not history: return ret
            if isinstance(history[0], list):
                return ret + ''.join([f'{self._soh}{h}{self._eoh}{self._soa}{a}{self._eoa}' for h, a in history])
            elif isinstance(history[0], dict):
                for item in history:
                    if item['role'] == 'user':
                        ret += f'{self._soh}{item["content"]}{self._eoh}'
                    elif item['role'] == 'assistant':
                        ret += f'{self._soa}'
                        ret += f'{item.get("content", "")}'
                        for idx in range(len(item.get('tool_calls', []))):
                            tool = item['tool_calls'][idx]['function']
                            if getattr(self, '_tool_args_token', None):
                                tool = tool['name'] + self._tool_args_token + \
                                    json.dumps(tool['arguments'], ensure_ascii=False)
                            ret += (f'{getattr(self, "_tool_start_token", "")}' + '\n'
                                    f'{tool}'
                                    f'{getattr(self, "_tool_end_token", "")}' + '\n')
                        ret += f'{self._eoa}'
                    elif item['role'] == 'tool':
                        try:
                            content = json.loads(item['content'].strip())
                        except Exception:
                            content = item['content']
                        ret += f'{getattr(self, "_soe", "")}{content}{getattr(self, "_eoe", "")}'

                return ret
            else:
                raise NotImplementedError('Cannot transform json history to {type(history[0])} now')

    def _get_instruction_and_input(self, input, *, return_dict=False, tools=None):
        instruction = self._instruction_template
        fc_prompt = '' if return_dict or not tools else FC_PROMPT
        if fc_prompt and FC_PROMPT_PLACEHOLDER not in instruction:
            instruction = f'{instruction}\n\n{fc_prompt}'
        instruction = instruction.replace(FC_PROMPT_PLACEHOLDER, fc_prompt)
        instruction = self._handle_tool_call_instruction(instruction, tools)
        prompt_keys = list(set(re.findall(r'\{(\w+)\}', instruction)))
        if isinstance(input, (str, int)):
            if len(prompt_keys) == 1:
                return instruction.format(**{prompt_keys[0]: input}), ''
            else:
                assert len(prompt_keys) == 0
                return instruction, input
        assert isinstance(input, dict), f'expected types are str, int and dict, bug get {type(input)}(`{input})`'
        kwargs = {k: input.pop(k) for k in prompt_keys}
        assert len(input) <= 1, f'Unexpected keys found in input: {list(input.keys())}'
        return (reduce(lambda s, kv: s.replace(f'{{{kv[0]}}}', kv[1]),
                       kwargs.items(),
                       instruction)
                if len(kwargs) > 0 else instruction,
                list(input.values())[0] if input else '')

    def _check_values(self, instruction, input, history, tools): pass

    # Used for TrainableModule(local deployed)
    def _generate_prompt_impl(self, instruction, input, user, history, tools, label):
        is_tool = False
        if isinstance(input, dict):
            input = input.get('content', '')
            is_tool = input.get('role') == 'tool'
        elif isinstance(input, list):
            is_tool = any(item.get('role') == 'tool' for item in input)
            input = '\n'.join([item.get('content', '') for item in input])
        params = dict(system=self._system, instruction=instruction, input=input, user=user, history=history, tools=tools,
                      sos=self._sos, eos=self._eos, soh=self._soh, eoh=self._eoh, soa=self._soa, eoa=self._eoa)
        if is_tool:
            params['soh'] = getattr(self, '_soe', self._soh)
            params['eoh'] = getattr(self, '_eoe', self._eoh)
        return self._template.format(**params) + (label if label else '')

    # Used for OnlineChatModule
    def _generate_prompt_dict_impl(self, instruction, input, user, history, tools, label):
        if not history: history = []
        if isinstance(input, str):
            history.append({'role': 'user', 'content': input})
        elif isinstance(input, dict):
            history.append(input)
        elif isinstance(input, list) and all(isinstance(ele, dict) for ele in input):
            history.extend(input)
        elif isinstance(input, tuple) and len(input) == 1:
            # Note tuple size 1 with one single string is not expected
            history.append({'role': 'user', 'content': input[0]})
        else:
            raise TypeError('input must be a string or a dict')

        if user:
            history[-1]['content'] = user + history[-1]['content']

        if self._enable_system:
            history.insert(0, {'role': 'system',
                               'content': self._system + '\n' + instruction if instruction else self._system})
        return dict(messages=history, tools=tools) if tools else dict(messages=history)

    def pre_hook(self, func: Optional[Callable] = None):
        """设置预处理钩子函数，供外部在生成提示词前对输入数据进行自定义处理。

Args:
    func (Optional[Callable]): 一个可调用对象，作为预处理钩子函数，接收并处理输入数据。

**Returns:**

- LazyLLMPrompterBase: 返回自身实例，方便链式调用。
"""
        self._pre_hook = func
        return self

    def _split_instruction(self, instruction: str):
        system_instruction = instruction
        user_instruction = ''
        if LazyLLMPrompterBase.ISA in instruction and LazyLLMPrompterBase.ISE in instruction:
            # The instruction includes system prompts and/or user prompts
            pattern = re.compile(r'%s(.*)%s' % (LazyLLMPrompterBase.ISA, LazyLLMPrompterBase.ISE), re.DOTALL)
            ret = re.split(pattern, instruction)
            system_instruction = ret[0]
            user_instruction = ret[1]

        return system_instruction, user_instruction

    def generate_prompt(self, input: Union[str, List, Dict[str, str], None] = None,
                        history: List[Union[List[str], Dict[str, Any]]] = None,
                        tools: Union[List[Dict[str, Any]], None] = None,
                        label: Union[str, None] = None,
                        *, show: bool = False, return_dict: bool = False) -> Union[str, Dict]:
        """根据用户输入，生成对应的Prompt.

Args:
    input (Option[str | Dict]):  Prompter的输入，如果是dict，会填充到instruction的槽位中；如果是str，则会作为输入。
    history (Option[List[List | Dict]]): 历史对话，可以为 ``[[u, s], [u, s]]`` 或 openai的history格式，默认为None。
    tools (Option[List[Dict]]: 可以使用的工具合集，大模型用作FunctionCall时使用，默认为None
    label (Option[str]): 标签，训练或微调时使用，默认为None
    show (bool): 标志是否打印生成的Prompt，默认为False
    return_dict (bool): 标志是否返回dict，一般情况下使用 ``OnlineChatModule`` 时会设置为True。如果返回dict，则仅填充 ``instruction``。默认为False
"""
        input = copy.deepcopy(input)
        if self._pre_hook:
            input, history, tools, label = self._pre_hook(input, history, tools, label)
        tools = tools or self._tools
        instruction, input = self._get_instruction_and_input(input, return_dict=return_dict, tools=tools)
        history = self._get_histories(history, return_dict=return_dict)
        tools = self._get_tools(tools, return_dict=return_dict)
        self._check_values(instruction, input, history, tools)
        instruction, user_instruction = self._split_instruction(instruction)
        func = self._generate_prompt_dict_impl if return_dict else self._generate_prompt_impl
        result = func(instruction, input, user_instruction, history, tools, label)
        if self._show or show: LOG.info(result)
        return result

    def get_response(self, output: str, input: Union[str, None] = None) -> str:
        """用作对Prompt的截断，只保留有价值的输出

Args:
     output (str): 大模型的输出
     input (Option[[str]): 大模型的输入，若指定此参数，会将输出中包含输入的部分全部截断，默认为None
"""
        if input and output.startswith(input):
            return output[len(input):]
        return output if getattr(self, '_split', None) is None else output.split(self._split)[-1]

class EmptyPrompter(LazyLLMPrompterBase):
    """继承自 `LazyLLMPrompterBase` 的空提示生成器，用于直接返回原始输入。

该类不会对输入进行任何处理，适用于无需格式化的调试、测试或占位场景。


Examples:
    >>> from lazyllm.components.prompter import EmptyPrompter
    >>> prompter = EmptyPrompter()
    >>> prompter.generate_prompt("Hello LazyLLM")
    'Hello LazyLLM'
    >>> prompter.generate_prompt({"query": "Tell me a joke"})
    {'query': 'Tell me a joke'}
    >>> # Even with additional parameters, the input is returned unchanged
    >>> prompter.generate_prompt("No-op", history=[["Hi", "Hello"]], tools=[{"name": "search"}], label="debug")
    'No-op'
    """

    def generate_prompt(self, input, history=None, tools=None, label=None, show=False, return_dict=False):
        """直接返回输入的Prompt实现，继承自 `LazyLLMPrompterBase`。

该方法不会对输入做任何格式化操作，适用于调试、测试或占位场景。

Args:
    input (Any): 任意输入，作为Prompt返回。
    history (Option[List[List | Dict]]): 历史对话，可忽略，默认None。
    tools (Option[List[Dict]]): 工具参数，可忽略，默认None。
    label (Option[str]): 标签，可忽略，默认None。
    show (bool): 是否打印返回内容，默认为False。
"""
        if return_dict:
            return {'messages': [{'role': 'user', 'content': input}]}
        if self._show or show: LOG.info(input)
        return input
