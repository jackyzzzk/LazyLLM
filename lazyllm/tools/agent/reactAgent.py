from .base import LazyLLMAgentBase
from lazyllm import loop, once_wrapper
from .functionCall import FunctionCall
from typing import List, Any, Dict, Optional, Union
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER

INSTRUCTION = f'''
## Role
You are a **ReAct-style autonomous agent** designed to solve user tasks through an iterative closed loop:
Reason → Act → Observe → Reflect.

Your goal is to produce **accurate, efficient, and well-structured results** while making optimal use of available tools.

## Core Working Loop (ReAct)
You must repeatedly follow this loop until the task is fully solved:
### 1. Reason
- Understand the user's intent and constraints.
- Break complex tasks into manageable steps.
- Decide whether external tools are required.
- Plan the next best action.

### 2. Act
- Select **the most appropriate tool** if a tool is needed.
- Invoke a tool **only when it provides capabilities or information you do not already have**.
- Avoid speculative, redundant, or exploratory tool calls.

### 3. Observe
- Carefully examine the tool's output.
- Extract only information relevant to the current task.
- Ignore noise or irrelevant details.

### 4. Reflect
- Evaluate whether the obtained information is sufficient.
- If insufficient, refine the plan and continue the loop.
- If sufficient, prepare the final answer.

## Tool Usage Guidelines
- You have access to multiple tools.
- Before calling a tool, always consider:
  * *Is a tool strictly necessary at this step?*
  * *Which available tool is the most suitable for this purpose?*
- Use **at most one tool per action step**.
- Do **not** call any tools after you already have enough information to answer.

{FC_PROMPT_PLACEHOLDER}

## Language & Communication Rules
- Always respond in **the same language as the user's question**.
- Do not switch languages unless explicitly requested.
- Be concise, precise, and task-focused.
- Do **not** expose internal reasoning, chain-of-thought, or decision deliberations to the user.

## Final Answer Rule
When the task is complete:
  * Stop the ReAct loop.
  * Do not call any additional tools.
  * Provide a clear and complete final answer.
The final response should contain **only the answer itself**, without internal process details.

You are responsible for maintaining correctness, efficiency, and clarity throughout the entire reasoning–action cycle.

'''


class ReactAgent(LazyLLMAgentBase):
    """ReactAgent是按照 `Thought->Action->Observation->Thought...->Finish` 的流程一步一步的通过LLM和工具调用来显示解决用户问题的步骤，以及最后给用户的答案。

Args:
    llm: 大语言模型实例，用于生成推理和工具调用决策
    tools (List[str]): 可用工具列表，可以是工具函数或工具名称
    max_retries (int): 最大重试次数，当工具调用失败时自动重试，默认为5
    return_trace (bool): 是否返回完整的执行轨迹，用于调试和分析，默认为False
    prompt (str): 自定义提示词模板，如果为None则使用内置模板
    stream (bool): 是否启用流式输出，用于实时显示生成过程，默认为False
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    desc (str): Agent 能力描述，可为空。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import fc_register, ReactAgent
    >>> @fc_register("tool")
    >>> def multiply_tool(a: int, b: int) -> int:
    ...     '''
    ...     Multiply two integers and return the result integer
    ...
    ...     Args:
    ...         a (int): multiplier
    ...         b (int): multiplier
    ...     '''
    ...     return a * b
    ...
    >>> @fc_register("tool")
    >>> def add_tool(a: int, b: int):
    ...     '''
    ...     Add two integers and returns the result integer
    ...
    ...     Args:
    ...         a (int): addend
    ...         b (int): addend
    ...     '''
    ...     return a + b
    ...
    >>> tools = ["multiply_tool", "add_tool"]
    >>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()   # or llm = lazyllm.OnlineChatModule(source="sensenova")
    >>> agent = ReactAgent(llm, tools)
    >>> query = "What is 20+(2*4)? Calculate step by step."
    >>> res = agent(query)
    >>> print(res)
    'Answer: The result of 20+(2*4) is 28.'
    """
    def __init__(self, llm, tools: Optional[List[str]] = None, max_retries: int = 5, return_trace: bool = False,
                 prompt: str = None, stream: bool = False, return_last_tool_calls: bool = False,
                 skills: Optional[Union[bool, str, List[str]]] = None, desc: str = '',
                 workspace: Optional[str] = None):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries, return_trace=return_trace,
                         stream=stream, return_last_tool_calls=return_last_tool_calls, skills=skills,
                         desc=desc, workspace=workspace)
        prompt = prompt or INSTRUCTION
        if self._return_last_tool_calls:
            prompt += '\nIf no more tool calls are needed, reply with ok and skip any summary.'
        assert self._llm is not None, 'llm cannot be empty.'
        self._assert_tools()
        self._prompt = prompt

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        agent = loop(FunctionCall(llm=self._llm, _prompt=self._prompt, return_trace=self._return_trace,
                                  stream=self._stream, _tool_manager=self._tools_manager,
                                  skill_manager=self._skill_manager, workspace=self.workspace),
                     stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._agent = agent

    def _pre_process(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        return (self._wrap_user_input_with_skills(query), llm_chat_history or [])

    def _post_process(self, ret):
        if isinstance(ret, str):
            completed = self._pop_tool_calls()
            if completed is not None:
                return completed
            return ret
        raise ValueError(f'After retrying {self._max_retries} times, the react agent still failes to call '
                         f'successfully.')
