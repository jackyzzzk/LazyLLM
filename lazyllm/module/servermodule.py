import re
import time
import json
import hashlib
import requests
import uuid
import pickle
import codecs
from typing import Callable, Dict, List, Union, Optional, Tuple
import copy
from dataclasses import dataclass

import lazyllm
from lazyllm import launchers, LOG, package, obj2str, globals, is_valid_url, LazyLLMLaunchersBase, redis_client
from lazyllm.common import _register_trim_module, _get_callsite
from ..components.formatter import FormatterBase, EmptyFormatter, decode_query_with_filepaths
from ..components.formatter.formatterbase import LAZYLLM_QUERY_PREFIX, _lazyllm_get_file_list
from ..components.prompter import PrompterBase, ChatPrompter, EmptyPrompter
from ..components.utils import LLMType
from ..flow import FlowBase, Pipeline
from urllib.parse import urljoin
from .utils import light_reduce
from .module import ModuleBase, ActionModule


_register_trim_module({'lazyllm.module.servermodule': ['__call__']})


class LLMBase(object):
    """大语言模型模块的基类，继承自 ModuleBase。  
负责管理流式输出、Prompt 和格式化器的初始化与切换，处理输入中的文件信息，支持实例共享。

Args:
    stream (bool 或 dict): 是否启用流式输出或流式配置，默认为 False。
    return_trace (bool): 是否返回执行过程的 trace，默认为 False。
    init_prompt (bool): 是否在初始化时自动创建默认 Prompt，默认为 True。
"""
    def __init__(self, stream: Union[bool, Dict[str, str]] = False,
                 init_prompt: bool = True, type: Optional[Union[str, LLMType]] = None):
        self._stream = stream
        self._type = LLMType(type) if type else LLMType.LLM
        if init_prompt: self.prompt()
        __class__.formatter(self)

    def _get_files(self, input, lazyllm_files):
        if isinstance(input, package):
            assert not lazyllm_files, 'Duplicate `files` argument provided by args and kwargs'
            input, lazyllm_files = input
        if isinstance(input, str) and input.startswith(LAZYLLM_QUERY_PREFIX):
            assert not lazyllm_files, 'Argument `files` is already provided by query'
            deinput = decode_query_with_filepaths(input)
            assert isinstance(deinput, dict), 'decode_query_with_filepaths must return a dict.'
            input, files = deinput['query'], deinput['files']
        else:
            files = _lazyllm_get_file_list(lazyllm_files) if lazyllm_files else []
        return input, files

    def prompt(self, prompt: Optional[str] = None, history: Optional[List[List[str]]] = None):
        """设置或切换 Prompt。支持 None、PrompterBase 子类或字符串/字典类型创建 ChatPrompter。

Args:
    prompt (str/dict/PrompterBase/None): 要设置的 Prompt。
    history (list): 对话历史，仅当 prompt 为字符串或字典时有效。

**Returns:**

- self: 便于链式调用。
"""
        if prompt is None:
            assert not history, 'history is not supported in EmptyPrompter'
            self._prompt = EmptyPrompter()
        elif isinstance(prompt, PrompterBase):
            assert not history, 'history is not supported in user defined prompter'
            self._prompt = prompt
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt, history=history)
        else:
            raise TypeError(f'{prompt} type is not supported.')
        return self

    def formatter(self, format: Optional[FormatterBase] = None):
        """设置或切换输出格式化器。支持 None、FormatterBase 子类或可调用对象。

Args:
    format (FormatterBase/Callable/None): 格式化器对象或函数，默认为 None。

**Returns:**

- self: 便于链式调用。
"""
        assert format is None or isinstance(format, FormatterBase) or callable(format), 'format must be None or Callable'
        self._formatter = format or EmptyFormatter()
        return self

    def share(self, prompt: Optional[Union[str, dict, PrompterBase]] = None, format: Optional[FormatterBase] = None,
              stream: Optional[Union[bool, Dict[str, str]]] = None, history: Optional[List[List[str]]] = None):
        """创建当前实例的浅拷贝，并可重新设置 prompt、formatter、stream 等属性。  
适用于多会话或多 Agent 共享基础配置但个性化部分参数的场景。

Args:
    prompt (str/dict/PrompterBase/None): 新的 Prompt，可选。
    format (FormatterBase/None): 新的格式化器，可选。
    stream (bool/dict/None): 新的流式设置，可选。
    history (list/None): 新的对话历史，仅在设置 Prompt 时有效。

**Returns:**

- LLMBase: 新的共享实例。
"""
        new = copy.copy(self)
        new._hooks = set()
        new._set_mid()
        if prompt is not None: new.prompt(prompt, history=history)
        if format is not None: new.formatter(format)
        if stream is not None: new.stream = stream
        return new

    @property
    def type(self):
        return self._type.value

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, v: Union[bool, Dict[str, str]]):
        self._stream = v

    def __or__(self, other):
        if not isinstance(other, FormatterBase):
            return NotImplemented
        return self.share(format=(other if isinstance(self._formatter, EmptyFormatter) else (self._formatter | other)))

    @property
    def appendix_hash_key(self):
        try:
            prompts = self._prompt.generate_prompt('x')
        except Exception:
            prompts = self._prompt._instruction_template
        if not isinstance(prompts, str):
            try:
                content = json.dumps(prompts, sort_keys=True)
            except Exception:
                content = str(prompts)
        else:
            content = prompts
        return hashlib.md5(content.encode()).hexdigest()

class _UrlHelper(object):
    @dataclass
    class _Wrapper:
        url: Optional[str] = None

    def __init__(self, url):
        self._url_wrapper = url if isinstance(url, _UrlHelper._Wrapper) else _UrlHelper._Wrapper(url=url)

    _url_id = property(lambda self: self._module_id)

    @property
    def _url(self) -> str:
        if not self._url_wrapper.url:
            if redis_client:
                try:
                    while not self._url_wrapper.url:
                        url = redis_client['url'].get(self._url_id)
                        self._url_wrapper.url = url.decode('utf-8') if url else None
                        if self._url_wrapper.url: break
                        time.sleep(lazyllm.config['redis_recheck_delay'])
                except Exception as e:
                    LOG.error(f'Error accessing Redis: {e}')
                    raise
        return self._url_wrapper.url

    def _set_url(self, url):
        if redis_client:
            redis_client['url'].set(self._url_id, url)
        LOG.debug(f'url: {url}')
        self._url_wrapper.url = url

    def _release_url(self):
        if redis_client:
            redis_client['url'].delete(self._url_id)

class UrlModule(ModuleBase, LLMBase, _UrlHelper):
    """可以将ServerModule部署得到的Url包装成一个Module，调用 ``__call__`` 时会访问该服务。

Args:
    url (str): 要包装的服务的Url，默认为空字符串
    stream (bool|Dict[str, str]): 是否流式请求和输出，默认为非流式
    return_trace (bool): 是否将结果记录在trace中，默认为False
    init_prompt (bool): 是否初始化prompt，默认为True


Examples:
    >>> import lazyllm
    >>> def demo(input): return input * 2
    ... 
    >>> s = lazyllm.ServerModule(demo, launcher=lazyllm.launchers.empty(sync=False))
    >>> s.start()
    INFO:     Uvicorn running on http://0.0.0.0:35485
    >>> u = lazyllm.UrlModule(url=s._url)
    >>> print(u(1))
    2
    """

    def __new__(cls, *args, **kw):
        if cls is not UrlModule:
            return super().__new__(cls)
        return ServerModule(*args, **kw)

    def __init__(self, *, url: Optional[str] = '', stream: Union[bool, Dict[str, str]] = False,
                 return_trace: bool = False, init_prompt: bool = True):
        super().__init__(return_trace=return_trace)
        LLMBase.__init__(self, stream=stream, init_prompt=init_prompt)
        _UrlHelper.__init__(self, url)

    def _estimate_token_usage(self, text):
        if not isinstance(text, str):
            return 0
        # extract english words, number and comma
        pattern = r'\b[a-zA-Z0-9]+\b|,'
        ascii_words = re.findall(pattern, text)
        ascii_ch_count = sum(len(ele) for ele in ascii_words)
        non_ascii_pattern = r'[^\x00-\x7F]'
        non_ascii_chars = re.findall(non_ascii_pattern, text)
        non_ascii_char_count = len(non_ascii_chars)
        return int(ascii_ch_count / 3.0 + non_ascii_char_count + 1)

    def _decode_line(self, line: bytes):
        try:
            return pickle.loads(codecs.decode(line, 'base64'))
        except Exception:
            return line.decode('utf-8')

    def _extract_and_format(self, output: str) -> str:
        return output

    def forward(self, *args, **kw):
        """定义了每次执行的计算步骤，ModuleBase的所有的子类都需要重写这个函数。


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...    def forward(self, input):
    ...        return input + 1
    ...
    >>> MyModule()(1)
    2
    """
        raise NotImplementedError

    def __call__(self, *args, **kw):
        assert self._url is not None, f'Please start {self.__class__} first'
        if len(args) > 1:
            return super(__class__, self).__call__(package(args), **kw)
        return super(__class__, self).__call__(*args, **kw)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Url', name=self._module_name, url=self._url,
                                 stream=self._stream, return_trace=self._return_trace)


@light_reduce
class _ServerModuleImpl(ModuleBase, _UrlHelper):
    def __init__(self, m=None, pre=None, post=None, launcher=None, port=None, pythonpath=None, url_wrapper=None,
                 num_replicas: int = 1, *, security_key: Optional[str] = None):
        super().__init__()
        _UrlHelper.__init__(self, url=url_wrapper)
        self._m = ActionModule(m) if isinstance(m, FlowBase) else m
        self._pre_func, self._post_func = pre, post
        self._launcher = launcher.clone() if launcher else launchers.remote(sync=False, ngpus=0)
        self._port = port
        self._pythonpath = pythonpath
        self._num_replicas = num_replicas
        self._security_key = security_key
        self._defined_pos = _get_callsite(depth=3)

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._m is None: return None
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self._m, pre_func=self._pre_func, port=self._port,
                                       pythonpath=self._pythonpath, post_func=self._post_func,
                                       launcher=self._launcher, num_replicas=self._num_replicas,
                                       security_key=self._security_key, defined_pos=self._defined_pos),
            self._set_url)

    def stop(self):
        self._launcher.cleanup()
        self._get_deploy_tasks.flag.reset()

    def __del__(self):
        self.stop()

    def __deepcopy__(self, memo):
        return self

class ServerModule(UrlModule):
    """ServerModule 类，继承自 UrlModule，封装了将任意可调用对象部署为 API 服务的能力。  
通过 FastAPI 实现，可以启动一个主服务和多个卫星服务，并支持流式调用、预处理和后处理逻辑。  
既可以传入本地可调用对象启动服务，也可以通过 URL 直接连接远程服务。

Args:
    m (Optional[Union[str, ModuleBase]]): 被包装成服务的模块或其名称。若为字符串则表示 URL，此时 `url` 必须为 None；若为 ModuleBase 则包装为服务。
    pre (Optional[Callable]): 前处理函数，在服务进程执行，默认为 ``None``。
    post (Optional[Callable]): 后处理函数，在服务进程执行，默认为 ``None``。
    stream (Union[bool, Dict]): 是否开启流式输出。可以是布尔值，或包含流式配置的字典，默认为 ``False``。
    return_trace (Optional[bool]): 是否返回调试追踪信息。默认为 ``False``。
    port (Optional[int]): 指定服务部署的端口。默认为 ``None``，将自动分配端口。
    pythonpath (Optional[str]): 传递给子进程的 PYTHONPATH 环境变量，默认为 ``None``。
    launcher (Optional[LazyLLMLaunchersBase]): 启动服务所使用的 Launcher，默认使用异步远程部署。
    url (Optional[str]): 已部署服务的 URL 地址。若提供，则 `m` 必须为 None。


Examples:
    >>> import lazyllm
    >>> def demo(input): return input * 2
    ...
    >>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
    >>> s.start()
    INFO:     Uvicorn running on http://0.0.0.0:35485
    >>> print(s(1))
    2

    >>> class MyServe(object):
    ...     def __call__(self, input):
    ...         return 2 * input
    ...
    ...     @lazyllm.FastapiApp.post
    ...     def server1(self, input):
    ...         return f'reply for {input}'
    ...
    ...     @lazyllm.FastapiApp.get
    ...     def server2(self):
    ...        return f'get method'
    ...
    >>> m = lazyllm.ServerModule(MyServe(), launcher=launchers.empty(sync=False))
    >>> m.start()
    INFO:     Uvicorn running on http://0.0.0.0:32028
    >>> print(m(1))
    2
    """
    def __init__(self, m: Optional[Union[str, ModuleBase]] = None, pre: Optional[Callable] = None,
                 post: Optional[Callable] = None, stream: Union[bool, Dict] = False,
                 return_trace: bool = False, port: Optional[int] = None, pythonpath: Optional[str] = None,
                 launcher: Optional[LazyLLMLaunchersBase] = None, url: Optional[str] = None,
                 num_replicas: int = 1, security_key: Optional[Union[str, bool]] = None):
        assert stream is False or return_trace is False, 'Module with stream output has no trace'
        assert (post is None) or (stream is False), 'Stream cannot be true when post-action exists'
        if isinstance(m, str):
            assert url is None, 'url should be None when m is a url'
            url, m = m, None
        if url:
            assert is_valid_url(url), f'Invalid url: {url}'
            assert m is None, 'm should be None when url is provided'
        super().__init__(url=url, stream=stream, return_trace=return_trace)
        self._security_key = f'sk-{str(uuid.uuid4().hex)}' if security_key is True else security_key
        self._impl = _ServerModuleImpl(m, pre, post, launcher, port, pythonpath, self._url_wrapper,
                                       num_replicas=num_replicas, security_key=self._security_key)
        if url: self._impl._get_deploy_tasks.flag.set()

    _url_id = property(lambda self: self._impl._module_id)

    def wait(self):
        """等待当前模块服务的启动或执行过程完成。  
通常用于阻塞主线程，直到服务正常结束或中断。  
"""
        self._impl._launcher.wait()

    def stop(self):
        """停止当前模块服务以及其相关子进程。  
调用后，模块将不再响应请求。  
"""
        self._impl.stop()

    @property
    def status(self):
        return self._impl._launcher.status

    def _call(self, fname, *args, **kwargs):
        args, kwargs = lazyllm.dump_obj(args), lazyllm.dump_obj(kwargs)
        url = urljoin(self._url.rsplit('/', 1)[0], '_call')
        r = requests.post(url, json=(fname, args, kwargs), headers={'Content-Type': 'application/json'})
        if r.status_code != 200:
            try:
                error_info = r.json()
            except ValueError:
                error_info = r.text
            raise requests.RequestException(f'{r.status_code}: {error_info}')
        return pickle.loads(codecs.decode(r.content, 'base64'))

    def forward(self, __input: Union[Tuple[Union[str, Dict], str], str, Dict] = package(), **kw):  # noqa B008
        headers = {
            'Content-Type': 'application/json',
            'Global-Parameters': globals.pickled_data,
            'Session-ID': globals._sid,
            'Security-Key': self._security_key,
        }
        data = obj2str((__input, kw))

        # context bug with httpx, so we use requests
        with requests.post(self._url, json=data, stream=True, headers=headers,
                           proxies={'http': None, 'https': None}) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            messages = ''
            with self.stream_output(self._stream):
                for line in r.iter_lines(delimiter=b'<|lazyllm_delimiter|>'):
                    line = self._decode_line(line)
                    if self._stream:
                        self._stream_output(str(line), getattr(self._stream, 'get', lambda x: None)('color'))
                    messages = (messages + str(line)) if self._stream else line

                temp_output = self._extract_and_format(messages)
                return self._formatter(temp_output)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Server', subs=[repr(self._impl._m)], name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)
