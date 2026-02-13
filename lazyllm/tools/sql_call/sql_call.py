from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm.tools.utils import chat_history_to_str
from lazyllm import pipeline, globals, bind, _0, switch
from typing import List, Any, Dict, Optional, Union, Callable
import datetime
import re
from lazyllm.tools.sql import DBManager

from ..rag import Document

sql_query_instruct_template = '''
Given the following SQL tables and current date {current_date}, your job is to write sql queries in {db_type} given a user’s request.

{desc}

Alert: Just reply the sql query in a code block start with triple-backticks and keyword 'sql'
'''  # noqa E501


mongodb_query_instruct_template = '''
Current date is {current_date}.
You are a seasoned expert with 10 years of experience in crafting NoSQL queries for {db_type}. 
I will provide a collection description in a specified format. 
Your task is to analyze the user_question, which follows certain guidelines, and generate a NoSQL MongoDB aggregation pipeline accordingly.

{desc}

Note: Please return the json pipeline in a code block start with triple-backticks and keyword 'json'.
'''  # noqa E501

db_explain_instruct_template = '''
According to chat history
```
{{history_info}}
```

and the {db_type} database description
```
{{desc}}
```

bellowing {statement_type} is executed

```
{{query}}
```
the result is
```
{{result}}
```
'''


class SqlCall(ModuleBase):
    """SqlCall 是一个扩展自 ModuleBase 的类,提供了使用语言模型(LLM)生成和执行 SQL 查询的接口。
它设计用于与 SQL 数据库交互,从语言模型的响应中提取 SQL 查询,执行这些查询,并返回结果或解释。

Args:
    llm: 用于生成和解释 SQL 查询及解释的大语言模型。
    sql_manager (DBManager): 数据库管理器实例，包含数据库连接和描述信息
    sql_examples (str, optional): SQL示例字符串，用于提示工程。默认为空字符串
    sql_post_func (Callable, optional): 对生成的SQL语句进行后处理的函数。默认为 ``None``
    use_llm_for_sql_result (bool, optional): 是否使用LLM来解释SQL执行结果。默认为 ``True``
    return_trace (bool, optional): 是否返回执行跟踪信息。默认为 ``False``


Examples:
        >>> # First, run SqlManager example
        >>> import lazyllm
        >>> from lazyllm.tools import SQLManger, SqlCall
        >>> sql_tool = SQLManger("personal.db")
        >>> sql_llm = lazyllm.OnlineChatModule(model="gpt-4o", source="openai", base_url="***")
        >>> sql_call = SqlCall(sql_llm, sql_tool, use_llm_for_sql_result=True)
        >>> print(sql_call("去年一整年销售额最多的员工是谁?"))
    """
    EXAMPLE_TITLE = 'Here are some example: '

    def __init__(self, llm, sql_manager: DBManager, sql_examples: str = '', sql_post_func: Callable = None,
                 use_llm_for_sql_result=True, return_trace: bool = False) -> None:
        super().__init__(return_trace=return_trace)
        if not sql_manager.desc:
            raise ValueError('Error: sql_manager found empty description.')
        self._sql_tool = sql_manager
        self.sql_post_func = sql_post_func

        if sql_manager.db_type == 'mongodb':
            self._query_prompter = ChatPrompter(instruction=mongodb_query_instruct_template).pre_hook(
                self.sql_query_promt_hook
            )
            statement_type = 'mongodb json pipeline'
            self._pattern = re.compile(r'```json(.+?)```', re.DOTALL)
        else:
            self._query_prompter = ChatPrompter(instruction=sql_query_instruct_template).pre_hook(
                self.sql_query_promt_hook
            )
            statement_type = 'sql query'
            self._pattern = re.compile(r'```sql(.+?)```', re.DOTALL)

        self._llm_query = llm.share(prompt=self._query_prompter).used_by(self._module_id)
        self._answer_prompter = ChatPrompter(
            instruction=db_explain_instruct_template.format(statement_type=statement_type, db_type=sql_manager.db_type)
        ).pre_hook(self.sql_explain_prompt_hook)
        self._llm_answer = llm.share(prompt=self._answer_prompter).used_by(self._module_id)
        self.example = sql_examples
        with pipeline() as sql_execute_ppl:
            sql_execute_ppl.exec = self._sql_tool.execute_query
            if use_llm_for_sql_result:
                sql_execute_ppl.concate = (lambda q, r: [q, r]) | bind(sql_execute_ppl.input, _0)
                sql_execute_ppl.llm_answer = self._llm_answer
        with pipeline() as ppl:
            ppl.llm_query = self._llm_query
            ppl.sql_extractor = self.extract_sql_from_response
            with switch(judge_on_full_input=False) as ppl.sw:
                ppl.sw.case[False, lambda x: x]
                ppl.sw.case[True, sql_execute_ppl]
        self._impl = ppl

    def sql_query_promt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: Optional[List[Union[List[str], Dict[str, Any]]]] = None,
        tools: Union[List[Dict[str, Any]], None] = None,
        label: Union[str, None] = None,
    ):
        """为从用户输入生成数据库查询准备 prompt 的 hook。

Args:
    input (Union[str, List, Dict[str, str], None]): 用户的自然语言查询。
    history (List[Union[List[str], Dict[str, Any]]]): 会话历史。
    tools (Union[List[Dict[str, Any]], None]): 可用工具描述。
    label (Union[str, None]): 可选标签。

**Returns:**

- Tuple: 包含格式化后的 prompt 字典（包括 current_date、db_type、desc、user_query）、history、tools 和 label。
"""
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        schema_desc = self._sql_tool.desc
        if self.example:
            schema_desc += f'\n{self.EXAMPLE_TITLE}\n{self.example}\n'
        if not isinstance(input, str):
            raise ValueError(f'Unexpected type for input: {type(input)}')
        return (
            dict(current_date=current_date, db_type=self._sql_tool.db_type, desc=schema_desc, user_query=input),
            history or [],
            tools,
            label,
        )

    def sql_explain_prompt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: List[Union[List[str], Dict[str, Any]]] = [],  # noqa B006
        tools: Union[List[Dict[str, Any]], None] = None,
        label: Union[str, None] = None,
    ):
        """为解释数据库查询执行结果准备 prompt 的 hook。

Args:
    input (Union[str, List, Dict[str, str], None]): 包含查询和结果的列表。
    history (List[Union[List[str], Dict[str, Any]]]): 会话历史。
    tools (Union[List[Dict[str, Any]], None]): 可用工具描述。
    label (Union[str, None]): 可选标签。

**Returns:**

- Tuple: 包含格式化后的 prompt 字典（history_info、desc、query、result、explain_query）、history、tools 和 label。
"""
        explain_query = 'Tell the user based on the execution results, making sure to keep the language consistent \
            with the user\'s input and don\'t translate original result.'
        if not isinstance(input, list) and len(input) != 2:
            raise ValueError(f'Unexpected type for input: {type(input)}')
        assert 'root_input' in globals and self._llm_answer._module_id in globals['root_input']
        user_query = globals['root_input'][self._llm_answer._module_id]
        globals.pop('root_input')
        history_info = chat_history_to_str(history, user_query)
        return (
            dict(
                history_info=history_info,
                desc=self._sql_tool.desc,
                query=input[0],
                result=input[1],
                explain_query=explain_query,
            ),
            history,
            tools,
            label,
        )

    def extract_sql_from_response(self, str_response: str) -> tuple[bool, str]:
        """从原始 LLM 响应中提取 SQL（或 MongoDB pipeline）语句。

Args:
    str_response (str): LLM 返回的原始文本，可能包含代码块。

**Returns:**

- tuple[bool, str]: 第一个元素表示是否成功提取，第二个是清洗后的或原始内容。如果提供了 sql_post_func，则会应用于提取结果。
"""
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            if self._sql_tool.db_type != 'mongodb':
                stmts = [s.strip() for s in extracted_content.split(';') if s.strip()]
                if stmts:
                    extracted_content = stmts[0]
            return True, extracted_content if not self.sql_post_func else self.sql_post_func(extracted_content)
        else:
            return False, str_response

    @classmethod
    def create_from_document(cls, document: Document, llm=None, sql_examples: str = '',
                             sql_post_func: Callable = None, use_llm_for_sql_result=True,
                             return_trace: bool = False) -> 'SqlCall':
        """基于已绑定 SchemaExtractor 的 Document 创建 SqlCall，复用其 NL2SQL SqlManager 和 LLM，可直接面向文档注册的 schema 生成/执行 SQL。

Args:
    document (Document): 具备 SchemaExtractor 的文档实例。
    llm (optional): 覆盖用于 SQL 生成/结果说明的 LLM，默认复用文档的 LLM。
    sql_examples (str, optional): 追加在 schema 描述后的 few-shot 示例，指导 SQL 生成。
    sql_post_func (Callable, optional): 对提取的 SQL/管道做后处理的函数。
    use_llm_for_sql_result (bool, optional): 是否用 LLM 解释查询结果，默认 True。
    return_trace (bool, optional): 是否返回流水线 trace，默认 False。

**Returns:**

- SqlCall: 绑定到该 Document schema 表的 SqlCall 实例。
"""
        if not document._schema_extractor:
            raise ValueError('Only document with schema extractor can be used to create SqlCall.')
        sql_manager = document._impl._schema_extractor.sql_manager_for_nl2sql(algo_id=document._curr_group)
        llm = document._schema_extractor._llm if not llm else llm
        if not sql_manager:
            raise ValueError('Sql manager is not initialized in schema extractor.')
        if not llm:
            raise ValueError('LLM is not initialized in schema extractor.')
        return cls(llm=llm, sql_manager=sql_manager, sql_examples=sql_examples,
                   sql_post_func=sql_post_func, use_llm_for_sql_result=use_llm_for_sql_result,
                   return_trace=return_trace)

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        globals['root_input'] = {self._llm_answer._module_id: input}
        if self._module_id in globals['chat_history']:
            globals['chat_history'][self._llm_query._module_id] = globals['chat_history'][self._module_id]
        return self._impl(input)
