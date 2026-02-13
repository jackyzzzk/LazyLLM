import os
import socket
import requests
import json
from typing import Union

import lazyllm
from lazyllm import LOG
from lazyllm import ModuleBase, ServerModule
from lazyllm.thirdparty import gradio as gr
from lazyllm.flow import Pipeline


class WebUi:
    """基于 Gradio 的知识库文件管理 Web UI 工具类。

该类用于构建一个简单的 Web 界面，支持创建分组、上传文件、列出/删除分组或文件，并通过 RESTful API 与后端交互。支持快速集成与展示文件管理能力。

Args:
    base_url (str): 后端 API 服务的基础地址。
"""
    def __init__(self, base_url) -> None:
        self.base_url = base_url

    def basic_headers(self, content_type=True):
        """
生成通用的 HTTP 请求头。

Args:
    content_type (bool): 是否包含 Content-Type 头信息（默认为 True）。

**Returns:**

- dict: HTTP 请求头字典。
"""
        return {
            'accept': 'application/json',
            'Content-Type': 'application/json' if content_type else None,
        }

    def muti_headers(
        self,
    ):
        """
生成多部分表单的HTTP请求头。
用于文件上传等需要multipart/form-data格式的请求。

**Returns:**

- Dict: 返回包含accept头部的HTTP请求头字典。
"""
        return {'accept': 'application/json'}

    def post_request(self, url, data):
        """
发送 POST 请求。

Args:
    url (str): 请求地址。
    data (dict): 请求数据，将被转为 JSON。

**Returns:**

- dict: 响应结果的 JSON。
"""
        response = requests.post(
            url, headers=self.basic_headers(), data=json.dumps(data)
        )
        return response.json()

    def get_request(self, url):
        """
发送 GET 请求。

Args:
    url (str): 请求地址。

**Returns:**

- dict: 响应结果的 JSON。
"""
        response = requests.get(url, headers=self.basic_headers(False))
        return response.json()

    def new_group(self, group_name: str):
        """
创建新的文件分组。

Args:
    group_name (str): 分组名称。

**Returns:**

- str: 创建结果的提示信息。
"""
        response = requests.post(
            f'{self.base_url}/new_group?group_name={group_name}',
            headers=self.basic_headers(True),
        )
        return response.json()['msg']

    def delete_group(self, group_name: str):
        """
删除指定的文件分组。

Args:
    group_name (str): 分组名称。

**Returns:**

- str: 删除结果信息。
"""
        response = requests.post(
            f'{self.base_url}/delete_group?group_name={group_name}',
            headers=self.basic_headers(True),
        )
        return response.json()['msg']

    def list_groups(self):
        """
获取所有知识库分组列表。
向后台API发送请求，获取当前所有的知识库分组信息。

**Returns:**

- List: 返回分组名称列表。
"""
        response = requests.get(
            f'{self.base_url}/list_kb_groups', headers=self.basic_headers(False)
        )
        return response.json()['data']

    def upload_files(self, group_name: str, override: bool = True):
        """
向指定分组上传文件。

Args:
    group_name (str): 分组名称。
    override (bool): 是否覆盖已存在的文件（默认 True）。

**Returns:**

- Any: 后端返回的上传结果数据。
"""
        response = requests.post(
            f'{self.base_url}/upload_files?group_name={group_name}&override={override}',
            headers=self.basic_headers(True),
        )
        return response.json()['data']

    def list_files_in_group(self, group_name: str):
        """
列出指定分组下的所有文件。

Args:
    group_name (str): 分组名称。

**Returns:**

- List: 文件信息列表。
"""
        response = requests.get(
            f'{self.base_url}/list_files_in_group?group_name={group_name}&alive=True',
            headers=self.basic_headers(False),
        )
        return response.json()['data']

    def delete_file(self, group_name: str, file_ids: list[str]):
        """
从指定分组中删除文件。

Args:
    group_name (str): 分组名称。
    file_ids (List[str]): 要删除的文件 ID 列表。

**Returns:**

- str: 删除结果提示。
"""
        response = requests.post(
            f'{self.base_url}/delete_files_from_group',
            headers=self.basic_headers(True),
            json={'group_name': group_name, 'file_ids': file_ids}
        )
        return response.json()['msg']

    def gr_show_list(self, str_list: list, list_name: Union[str, list]):
        """
以 Gradio 表格的形式展示字符串列表。

Args:
    str_list (List): 字符串或子项列表。
    list_name (Union[str, List]): 表头名称或列名列表。

**Returns:**

- gr.DataFrame: Gradio 表格组件。
"""
        if isinstance(list_name, str):
            headers = ['index', list_name]
            value = [[index, str_list[index]] for index in range(len(str_list))]
        else:
            headers = ['index'] + list_name
            value = [[index] + str_list[index:index + len(list_name)] for index in range(len(str_list))]
        return gr.DataFrame(headers=headers, value=value)

    def create_ui(self):
        """
构建包含多个标签页的Gradio界面，提供以下功能：
    - 分组列表：查看所有分组信息
    - 上传文件：选择分组并上传文件
    - 分组文件列表：查看指定分组中的文件
    - 删除文件：从分组中删除指定文件

**Returns:**

- gr.Blocks: 完整的 Gradio UI 应用实例。
"""
        with gr.Blocks(analytics_enabled=False) as demo:
            with gr.Tabs():
                select_group_list = []

                with gr.TabItem('分组列表'):
                    select_group = self.gr_show_list(
                        self.list_groups(), list_name='group_name'
                    )
                    select_group_list.append(select_group)

                with gr.TabItem('上传文件'):

                    def _upload_files(group_name, files):

                        files_to_upload = [
                            ('files', (os.path.basename(file), open(file, 'rb')))
                            for file in files
                        ]

                        url = f'{self.base_url}/add_files_to_group?group_name={group_name}&override=true'
                        response = requests.post(
                            url, files=files_to_upload, headers=self.muti_headers()
                        )
                        response.raise_for_status()
                        response_data = response.json()
                        gr.Info(str(response_data['msg']))

                        for _, (_, file_obj) in files_to_upload:
                            file_obj.close()

                    select_group = gr.Dropdown(self.list_groups(), label='选择分组')
                    select_group.change(lambda x: x, inputs=select_group, outputs=None)

                    up_files = gr.Files(label='上传文件')
                    up_btn = gr.Button('上传')
                    up_btn.click(
                        _upload_files,
                        inputs=[select_group, up_files],
                        outputs=None,
                    )

                    select_group_list.append(select_group)

                with gr.TabItem('分组文件列表'):
                    def _list_group_files(group_name):
                        file_list = self.list_files_in_group(group_name)
                        values = [[i] + file_list[i][:2] for i in range(len(file_list))]
                        return gr.update(
                            value=values
                        )

                    select_group = gr.Dropdown(self.list_groups(), label='选择分组')
                    show_list = self.gr_show_list([], list_name=['file_id', 'file_name'])
                    select_group.change(
                        fn=_list_group_files, inputs=select_group, outputs=show_list
                    )
                    select_group_list.append(select_group)

                with gr.TabItem('删除文件'):

                    def _list_group_files(group_name):
                        file_list = self.list_files_in_group(group_name)
                        file_list = [','.join(file[:2]) for file in file_list]
                        return gr.update(choices=file_list)

                    select_group = gr.Dropdown(self.list_groups(), label='选择分组')
                    select_file = gr.Dropdown([], label='选择文件')
                    select_group.change(
                        fn=_list_group_files, inputs=select_group, outputs=select_file
                    )
                    delete_btn = gr.Button('删除')

                    def _delete_file(group_name, select_file):
                        file_ids = [select_file.split(',')[0]]
                        gr.Info(self.delete_file(group_name, file_ids))
                        return _list_group_files(group_name)

                    delete_btn.click(
                        fn=_delete_file,
                        inputs=[select_group, select_file],
                        outputs=select_file,
                    )
                    select_group_list.append(select_group)

        return demo


class DocWebModule(ModuleBase):
    """文档Web界面模块，继承自ModuleBase，提供基于Web的文档管理交互界面。

Args:
    doc_server (ServerModule): 文档服务模块实例，提供后端API支持
    title (str, optional): 界面标题，默认为"文档管理演示终端"
    port (optional): 服务端口号或端口范围。默认为 ``None``（使用20800-20999范围）
    history (optional): 初始聊天历史记录，默认为 ``None``
    text_mode (optional): 文本处理模式，默认为``None``(动态模式)
    trace_mode (optional): 追踪模式，默认为``None``(刷新模式)

类属性:

    Mode: 模式枚举类，包含:
        - Dynamic: 动态模式
        - Refresh: 刷新模式
        - Appendix: 附录模式

注意事项:
    - 需要配合有效的doc_server实例使用
    - 端口冲突时会自动尝试范围内其他端口
    - 服务停止后会释放相关资源


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools.rag.web import DocWebModule
    >>> from lazyllm import
    >>> doc_server = ServerModule(url="your_url")
    >>> doc_web = DocWebModule(
    >>>   doc_server=doc_server,
    >>>   title="文档管理演示终端",
    >>>   port=range(20800, 20805)  # 自动寻找可用端口)
    >>> deploy_task = doc_web._get_deploy_tasks()
    >>> deploy_task()
    >>> print(doc_web.url)
    >>> doc_web.stop()
    """
    class Mode:
        Dynamic = 0
        Refresh = 1
        Appendix = 2

    def __init__(self, doc_server: ServerModule, title='文档管理演示终端', port=None,
                 history=None, text_mode=None, trace_mode=None) -> None:
        super().__init__()
        self.title = title
        self.port = port or range(20800, 20999)
        self.history = history or []
        self.trace_mode = trace_mode if trace_mode else DocWebModule.Mode.Refresh
        self.text_mode = text_mode if text_mode else DocWebModule.Mode.Dynamic
        self.doc_server = doc_server
        self._deploy_flag = lazyllm.once_flag()
        self.api_url = ''
        self.url = ''

    def _prepare(self, query, chat_history):
        if chat_history is None:
            chat_history = []
        return '', chat_history + [[query, None]]

    def _clear_history(self):
        return [], '', ''

    def _work(self):
        if isinstance(self.port, (range, tuple, list)):
            port = self._find_can_use_network_port()
        else:
            port = self.port
            assert self._verify_port_access(port), f'port {port} is occupied'

        self.api_url = self.doc_server._url.rsplit('/', 1)[0]
        self.web_ui = WebUi(self.api_url)
        self.demo = self.web_ui.create_ui()
        self.url = f'http://127.0.0.1:{port}'
        self.broadcast_url = f'http://0.0.0.0:{port}'

        self.demo.queue().launch(server_name='0.0.0.0', server_port=port, prevent_thread_lock=True)
        LOG.success('LazyLLM docwebmodule launched successfully: Running on: '
                    f'{self.broadcast_url}, local URL: {self.url}')

    def _get_deploy_tasks(self):
        return Pipeline(self._work)

    def _get_post_process_tasks(self):
        return Pipeline(self._print_url)

    def wait(self):
        """阻塞当前线程等待Web服务运行。

该方法会阻塞调用线程，直到Web服务被显式停止。

"""
        self.demo.block_thread()

    def stop(self):
        """停止Web界面服务并释放相关资源。

"""
        if self.demo:
            self.demo.close()
            del self.demo
            self.demo, self.url = None, ''

    def _find_can_use_network_port(self):
        for port in self.port:
            if self._verify_port_access(port):
                return port
        raise RuntimeError(
            f'The ports in the range {self.port} are all occupied. '
            'Please change the port range or release the relevant ports.'
        )

    def _print_url(self):
        lazyllm.LOG.success(f'LazyLLM DocWebModule launched successfully: Running on local URL: {self.url}')

    def _verify_port_access(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('127.0.0.1', port))
            return result != 0

    def __repr__(self):
        return lazyllm.make_repr('Module', 'DocWebModule')
