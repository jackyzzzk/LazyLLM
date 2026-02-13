from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, loop, locals, Color, package, FileSystemQueue, colored_text, once_wrapper
from .toolsManager import ToolManager
from .skill_manager import SKILLS_PROMPT
from typing import List, Any, Dict, Union, Callable, Optional
from .base import LazyLLMAgentBase
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER
from lazyllm.common.deprecated import deprecated
import re
import json

FC_PROMPT = f'''# Tools

## You have access to the following tools:
## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs.
{FC_PROMPT_PLACEHOLDER}

Don\'t make assumptions about what values to plug into functions.
Ask for clarification if a user request is ambiguous.\n
'''


class StreamResponse():
    """StreamResponse类用于封装带有前缀和颜色配置的流式输出行为。
当启用流式模式时，调用实例会将带颜色的文本推送到文件系统队列中，用于异步处理或显示。

Args:
    prefix (str): 输出内容前的前缀文本，通常用于标识信息来源或类别。
    prefix_color (Optional[str]): 前缀文本的颜色，支持终端颜色代码，默认无颜色。
    color (Optional[str]): 主体内容文本颜色，支持终端颜色代码，默认无颜色。
    stream (bool): 是否启用流式输出模式，启用后会将文本推送至文件系统队列，默认关闭。


Examples:
    >>> from lazyllm.tools.agent.functionCall import StreamResponse
    >>> resp = StreamResponse(prefix="[INFO]", prefix_color="green", color="white", stream=True)
    >>> resp("Hello, world!")
    Hello, world!
    """
    def __init__(self, prefix: str, prefix_color: str = None, color: str = None, stream: bool = False):
        self.stream = stream
        self.prefix = prefix
        self.prefix_color = prefix_color
        self.color = color

    def __call__(self, *inputs):
        if self.stream: FileSystemQueue().enqueue(colored_text(f'\n{self.prefix}\n', self.prefix_color))
        if len(inputs) == 1:
            if self.stream: FileSystemQueue().enqueue(colored_text(f'{inputs[0]}', self.color))
            return inputs[0]
        if self.stream: FileSystemQueue().enqueue(colored_text(f'{inputs}', self.color))
        return package(*inputs)


class FunctionCall(ModuleBase):
    """FunctionCall是单轮工具调用类。当LLM自身信息不足以回答用户问题，需要结合外部工具获取辅助信息时，调用此类。
若LLM输出需要调用工具，则执行工具调用并返回调用结果；输出结果为List类型，包含当前轮的输入、模型输出和工具输出。
若不需工具调用，则直接返回LLM输出结果，输出为字符串类型。

Args:
    llm (ModuleBase): 使用的LLM实例，支持TrainableModule或OnlineChatModule。
    tools (List[Union[str, Callable]]): LLM可调用的工具名称或Callable对象列表。
    return_trace (Optional[bool]): 是否返回调用轨迹，默认为False。
    stream (Optional[bool]): 是否启用流式输出，默认为False。
    _prompt (Optional[str]): 自定义工具调用提示语，默认根据llm类型自动设置。

注意：tools中的工具需包含`__doc__`字段，且须遵循[Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)规范说明用途与参数。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import fc_register, FunctionCall
    >>> import json
    >>> from typing import Literal
    >>> @fc_register("tool")
    >>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"] = 'fahrenheit'):
    ...     '''
    ...     Get the current weather in a given location
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         unit (str): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> @fc_register("tool")
    >>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"] = 'fahrenheit'):
    ...     '''
    ...     Get an N-day weather forecast
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         num_days (int): The number of days to forecast.
    ...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> tools=["get_current_weather", "get_n_day_weather_forecast"]
    >>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule("openai", stream=False)
    >>> query = "What's the weather like today in celsius in Tokyo."
    >>> fc = FunctionCall(llm, tools)
    >>> ret = fc(query)
    >>> print(ret)
    ["What's the weather like today in celsius in Tokyo.", {'role': 'assistant', 'content': '
    ', 'tool_calls': [{'id': 'da19cddac0584869879deb1315356d2a', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo', 'unit': 'celsius'}}}]}, [{'role': 'tool', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'tool_call_id': 'da19cddac0584869879deb1315356d2a', 'name': 'get_current_weather'}]]
    >>> query = "Hello"
    >>> ret = fc(query)
    >>> print(ret)
    'Hello! How can I assist you today?'
    """

    def __init__(self, llm, tools: Optional[List[Union[str, Callable]]] = None, *, return_trace: bool = False,
                 stream: bool = False, _prompt: str = None, _tool_manager: Optional[ToolManager] = None,
                 skill_manager=None, workspace: Optional[str] = None):
        super().__init__(return_trace=return_trace)
        if _tool_manager is None:
            assert tools, 'tools cannot be empty.'
            self._tools_manager = ToolManager(tools, return_trace=return_trace)
        else:
            self._tools_manager = _tool_manager
        self._skill_manager = skill_manager
        self._workspace = workspace
        prompt = _prompt or FC_PROMPT
        if self._workspace:
            prompt = (
                f'{prompt}\n\n## Workspace\n'
                f'- Default workspace: `{self._workspace}`\n'
                '- Prefer creating/updating files under this workspace.\n'
                '- Use absolute paths under this workspace when creating files.\n'
            )
        if self._skill_manager:
            prompt = f'{prompt}\n\n{SKILLS_PROMPT}'
            self._prompter = ChatPrompter(
                instruction=prompt,
                tools=self._tools_manager.tools_description,
                extra_keys=['available_skills']
            )
        else:
            self._prompter = ChatPrompter(instruction=prompt, tools=self._tools_manager.tools_description)
        self._llm = llm.share(prompt=self._prompter, format=FunctionCallFormatter()).used_by(self._module_id)
        with pipeline() as self._impl:
            self._impl.ins = StreamResponse('Received instruction:', prefix_color=Color.yellow,
                                            color=Color.green, stream=stream)
            self._impl.pre_action = self._build_history
            self._impl.llm = self._llm
            self._impl.dis = StreamResponse('Decision-making or result in this round:',
                                            prefix_color=Color.yellow, color=Color.green, stream=stream)
            self._impl.post_action = self._post_action

    def _build_history(self, input: Union[str, dict, list]):
        workspace = locals['_lazyllm_agent']['workspace']
        history_idx = len(workspace.setdefault('history', []))
        if self._skill_manager and isinstance(input, dict) and input.get('available_skills'):
            workspace['available_skills'] = input['available_skills']
        if isinstance(input, str):
            workspace['history'].append({'role': 'user', 'content': input})
        elif isinstance(input, dict) and 'input' in input:
            workspace['history'].append(
                {'role': 'user', 'content': input.get('input', '')}
            )
        elif isinstance(input, dict) and input.get('role') == 'user':
            workspace['history'].append(
                {'role': 'user', 'content': input.get('content', '')}
            )
        elif isinstance(input, dict):
            tool_call_results = [
                {
                    'role': 'tool',
                    'content': str(tool_call['tool_call_result']),
                    'tool_call_id': tool_call['id'],
                    'name': tool_call['function']['name'],
                } for tool_call in workspace['tool_call_trace']
            ]
            workspace['history'].append(
                {'role': 'assistant', 'content': input.get('content', ''), 'tool_calls': input.get('tool_calls', [])}
            )
            input = {'input': tool_call_results}
            history_idx += 1
            workspace['history'].extend(tool_call_results)
        chat_history = workspace['history'][:history_idx]
        locals['chat_history'][self._llm._module_id] = chat_history
        if self._skill_manager and isinstance(input, dict) and 'available_skills' not in input:
            available = workspace.get('available_skills')
            if available: input['available_skills'] = available
        return input

    def _post_action(self, llm_output: Dict[str, Any]):
        if not llm_output.get('tool_calls'):
            if (match := re.search(r'Action:\s*Call\s+(\w+)\s+with\s+parameters\s+(\{.*?\})', llm_output['content'])):
                try:
                    llm_output['tool_calls'] = [{'function': {'name': match.group(1),
                                                              'arguments': json.loads(match.group(2))}}]
                except Exception: pass
        if tool_calls := llm_output.get('tool_calls'):
            if isinstance(tool_calls, list): [item.pop('index', None) for item in tool_calls]
            tool_calls_results = self._tools_manager(tool_calls)
            locals['_lazyllm_agent']['workspace']['tool_call_trace'] = [
                {**tool_call, 'tool_call_result': tool_result}
                for tool_call, tool_result in zip(tool_calls, tool_calls_results)
            ]
        else:
            llm_output = llm_output['content']
        return llm_output

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if 'workspace' not in locals['_lazyllm_agent']:
            locals['_lazyllm_agent']['workspace'] = dict(history=llm_chat_history or [])
        result = self._impl(input)

        # If the model decides not to call any tools, the result is a string. For debugging and subsequent tasks,
        # the last non-empty tool call trace is stored in locals['_lazyllm_agent']['completed'].
        if isinstance(result, str):
            locals['_lazyllm_agent']['completed'] = locals['_lazyllm_agent'].pop('workspace')\
                .pop('tool_call_trace', locals['_lazyllm_agent'].get('completed', []))
            locals['chat_history'][self._llm._module_id] = []
        return result

@deprecated('ReactAgent')
class FunctionCallAgent(LazyLLMAgentBase):
    """(FunctionCallAgent 已被废弃，将在未来版本中移除。请使用 ReactAgent 代替。) FunctionCallAgent是一个使用工具调用方式进行完整工具调用的代理，即回答用户问题时，LLM如果需要通过工具获取外部知识，就会调用工具，并将工具的返回结果反馈给LLM，最后由LLM进行汇总输出。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。
    tools (List[str]): LLM 使用的工具名称列表。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
    return_trace (bool): 是否返回执行追踪信息，默认为False。
    stream (bool): 是否启用流式输出，默认为False。
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    desc (str): Agent 能力描述，可为空。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import fc_register, FunctionCallAgent
    >>> import json
    >>> from typing import Literal
    >>> @fc_register("tool")
    >>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]='fahrenheit'):
    ...     '''
    ...     Get the current weather in a given location
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         unit (str): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    ...     elif 'beijing' in location.lower():
    ...         return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> @fc_register("tool")
    >>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
    ...     '''
    ...     Get an N-day weather forecast
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         num_days (int): The number of days to forecast.
    ...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
    ...     elif 'beijing' in location.lower():
    ...         return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> tools = ['get_current_weather', 'get_n_day_weather_forecast']
    >>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
    >>> agent = FunctionCallAgent(llm, tools)
    >>> query = "What's the weather like today in celsius in Tokyo and Paris."
    >>> res = agent(query)
    >>> print(res)
    'The current weather in Tokyo is 10 degrees Celsius, and in Paris, it is 22 degrees Celsius.'
    >>> query = "Hello"
    >>> res = agent(query)
    >>> print(res)
    'Hello! How can I assist you today?'
    """
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = '',
                 workspace: Optional[str] = None):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls,
                         skills=skills, desc=desc, workspace=workspace)
        assert self._llm is not None, 'llm cannot be empty.'
        self._assert_tools()
        prompt = FC_PROMPT
        self._fc = FunctionCall(llm=self._llm, return_trace=return_trace, stream=stream,
                                _prompt=prompt, _tool_manager=self._tools_manager,
                                skill_manager=self._skill_manager, workspace=self.workspace)
        self._fc._llm.used_by(self._module_id)

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        agent = loop(self._fc, stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._agent = agent

    def _pre_process(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        query = self._wrap_user_input_with_skills(query)
        if llm_chat_history is not None:
            return (query, llm_chat_history)
        return query

    def _post_process(self, ret):
        if isinstance(ret, str):
            completed = self._pop_tool_calls()
            if completed is not None:
                return completed
            return ret
        raise ValueError(f'After retrying {self._max_retries} times, the function call agent still fails to call '
                         f'successfully.')
