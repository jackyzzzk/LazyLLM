from lazyllm.module import ModuleBase
from .base import LazyLLMAgentBase
from lazyllm.components import ChatPrompter
from lazyllm import pipeline, LOG, bind, Color, locals, ifs, once_wrapper
from typing import Callable, Dict, List, Optional, Union
import re
import json

P_PROMPT_PREFIX = ('For the following tasks, make plans that can solve the problem step-by-step. '
                   'For each plan, indicate which external tool together with tool input to retrieve '
                   'evidence. You can store the evidence into a variable #E that can be called by '
                   'later tools. (Plan, #E1, Plan, #E2, Plan, #E3...) \n\n')

P_FEWSHOT = '''For example,
Task: We are planning to visit the capital city of China this week. What clothing should we wear for the trip?
Plan: First, search for the capital city of China.
#E1 = search[{"query": "What\'s the capital city of China?"}]
Plan: Next, obtain the weather forecast for this week in the capital city of China.
#E2 = weather[{"location": "#E1", "days": 7}]
Plan: Finally, use a language model to generate clothing recommendations based on the weekly weather.
#E3 = llm[{"input": "Using the 7-day forecast in #E2, provide clothing suggestions for visiting #E1 this week."}]'''

P_PROMPT_SUFFIX = '''Begin! Describe your plans with rich details. Each Plan should be followed by only one #E,
and the params_dict is the input of the tool, should be a valid json string wrapped in [],
(e.g. [{{'input': 'hello world', 'num_beams': 5}}]).\n\n'''

S_PROMPT_PREFIX = ('Solve the following task or problem. To assist you, we provide some plans and '
                   'corresponding evidences that might be helpful. Notice that some of these information '
                   'contain noise so you should trust them with caution.\n\n')

S_PROMPT_SUFFIX = ('\nNow begin to solve the task or problem. Respond with '
                   'the answer directly with no extra words.\n\n')
S_PROMPT_TEMPLATE = S_PROMPT_PREFIX + '{objective}\n{worker_evidences}' + S_PROMPT_SUFFIX + '{objective}\n'

class ReWOOAgent(LazyLLMAgentBase):
    """ReWOOAgent包含三个部分：Planner、Worker和Solver。其中，Planner使用可预见推理能力为复杂任务创建解决方案蓝图；Worker通过工具调用来与环境交互，并将实际证据或观察结果填充到指令中；Solver处理所有计划和证据以制定原始任务或问题的解决方案。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。和plan_llm、solve_llm互斥，要么设置llm(planner和solver公用一个LLM)，要么设置plan_llm和solve_llm，或者只指定llm(用来设置planner)和solve_llm，其它情况均认为是无效的。
    tools (List[str]): LLM使用的工具名称列表。
    plan_llm (ModuleBase): planner要使用的LLM，可以是TrainableModule或OnlineChatModule。
    solve_llm (ModuleBase): solver要使用的LLM，可以是TrainableModule或OnlineChatModule。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    desc (str): Agent 能力描述，可为空。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。



Examples:
    >>> import lazyllm
    >>> import wikipedia
    >>> from lazyllm.tools import fc_register, ReWOOAgent
    >>> @fc_register("tool")
    >>> def WikipediaWorker(input: str):
    ...     '''
    ...     Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge about people, places, companies, historical events, or other subjects. The response are long and might contain some irrelevant information. Input should be a search query.
    ...
    ...     Args:
    ...         input (str): search query.
    ...     '''
    ...     try:
    ...         evidence = wikipedia.page(input).content
    ...         evidence = evidence.split("\\n\\n")[0]
    ...     except wikipedia.PageError:
    ...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    ...     except wikipedia.DisambiguationError:
    ...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    ...     return evidence
    ...
    >>> @fc_register("tool")
    >>> def LLMWorker(input: str):
    ...     '''
    ...     A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.
    ...
    ...     Args:
    ...         input (str): instruction
    ...     '''
    ...     llm = lazyllm.OnlineChatModule(source="glm")
    ...     query = f"Respond in short directly with no extra words.\\n\\n{input}"
    ...     response = llm(query, llm_chat_history=[])
    ...     return response
    ...
    >>> tools = ["WikipediaWorker", "LLMWorker"]
    >>> llm = lazyllm.TrainableModule("GLM-4-9B-Chat").deploy_method(lazyllm.deploy.vllm).start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
    >>> agent = ReWOOAgent(llm, tools)
    >>> query = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"
    >>> res = agent(query)
    >>> print(res)
    '
    Hennessy '
    """
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[Union[str, Callable]] = [], *,  # noqa B006
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 return_trace: bool = False, stream: bool = False, return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = '',
                 workspace: Optional[str] = None):
        super().__init__(llm=llm, tools=tools, return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls, skills=skills, desc=desc,
                         workspace=workspace)
        if llm is None and plan_llm is None and solve_llm is None:
            raise ValueError('Either specify llm, or provide plan_llm/solve_llm.')
        if llm is None:
            if plan_llm is None:
                plan_llm = solve_llm
            if solve_llm is None:
                solve_llm = plan_llm
        else:
            if plan_llm is None:
                plan_llm = llm
            if solve_llm is None:
                solve_llm = llm
        self._assert_tools()
        extra_keys = ['available_skills'] if self._skill_manager else None
        planner_prompt = self._append_skills_prompt(self._build_planner_prompt_template())
        solver_prompt = self._append_workspace_prompt(
            self._append_skills_prompt(S_PROMPT_TEMPLATE)
        )
        self._planner = plan_llm.share(
            prompt=ChatPrompter(instruction=planner_prompt, extra_keys=extra_keys),
            stream=dict(prefix='\nI will give a plan first:\n', prefix_color=Color.blue, color=Color.green)
            if stream else False
        )
        self._solver = solve_llm.share(
            prompt=ChatPrompter(instruction=solver_prompt, extra_keys=extra_keys),
            stream=dict(prefix='\nI will solve the problem:\n', prefix_color=Color.blue, color=Color.green)
            if stream else False
        )

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        with pipeline() as agent:
            agent.planner_pre_action = self._build_planner_input
            agent.planner = self._planner
            agent.worker_evidences = self._get_worker_evidences
            agent.solver_pre_action = self._build_solver_input | bind(input=agent.input)
            agent.solver = ifs(self._return_last_tool_calls, lambda x: 'ok', self._solver)
        self._agent = agent

    def _build_planner_prompt_template(self):
        prompt = P_PROMPT_PREFIX + 'Tools can be one of the following:\n'
        for name, tool in self._tools_manager.tools_info.items():
            prompt += f'{name}[params_dict]: {tool.description}\n'
        prompt += P_FEWSHOT + '\n' + P_PROMPT_SUFFIX
        return prompt

    def _build_planner_input(self, input: str):
        locals['chat_history'][self._planner._module_id] = []
        return self._wrap_user_input_with_skills(input)

    def _parse_and_call_tool(self, tool_call: str, evidence: Dict[str, str]):
        tool_name, tool_arguments = tool_call.split('[', 1)
        tool_arguments = tool_arguments.split(']')[0]
        for var in re.findall(r'#E\d+', tool_arguments):
            if var in evidence:
                tool_arguments = tool_arguments.replace(var, str(evidence[var]))
        tool_calls = [{'function': {'name': tool_name, 'arguments': tool_arguments}}]
        result = self._tools_manager(tool_calls)
        locals['_lazyllm_agent']['workspace']['tool_call_trace'].append(
            {**tool_calls[0], 'tool_call_result': result[0]}
        )
        return json.dumps(result[0]).strip('\"')

    def _get_worker_evidences(self, response: str):
        LOG.debug(f'planner plans: {response}')
        evidence = {}
        worker_evidences = ''
        for line in response.splitlines():
            if line.startswith('Plan'):
                worker_evidences += line + '\n'
            elif re.match(r'#E\d+\s*=', line.strip()):
                e, tool_call = line.split('=', 1)
                evidence[e.strip()] = self._parse_and_call_tool(tool_call.strip(), evidence)
                worker_evidences += f'Evidence:\n{evidence[e.strip()]}\n'

        LOG.debug(f'worker_evidences: {worker_evidences}')
        return worker_evidences

    def _build_solver_input(self, worker_evidences, input):
        locals['chat_history'][self._solver._module_id] = []
        payload = {'input': input, 'objective': input, 'worker_evidences': worker_evidences}
        if self._skill_manager:
            return self._skill_manager.wrap_input(payload, input)
        return payload

    def _pre_process(self, query: str):
        locals['_lazyllm_agent']['workspace'] = {'tool_call_trace': []}
        return query

    def _post_process(self, result):
        trace = self._pop_tool_calls()
        if trace is not None:
            return trace
        return result
