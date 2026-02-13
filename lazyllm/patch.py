import os
import sys
import inspect
import requests
import ipaddress
import importlib.abc
import importlib.util
from typing import Callable
from urllib.parse import urlparse

def _is_ip_address_url(url: str) -> bool:
    try:
        hostname = urlparse(url).hostname
        if hostname is None:
            return False
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False

no_proxies = set(os.environ.get('no_proxy', '').split(','))
no_proxies.update({'localhost', '127.0.0.1', 'localaddress', '.localdomain.com'})
os.environ['no_proxy'] = ','.join(list(no_proxies))


def request(method, url, **kwargs):
    with requests.sessions.Session() as session:
        if os.environ.get('http_proxy') and _is_ip_address_url(url):
            try:
                session.trust_env = False
                return session.request(method=method, url=url, **kwargs)
            except Exception: pass
        session.trust_env = True
        return session.request(method=method, url=url, **kwargs)

def _get(url, params=None, **kwargs): return request('get', url, params=params, **kwargs)
def _options(url, **kwargs): return request('options', url, **kwargs)
def _post(url, data=None, json=None, **kwargs): return request('post', url, data=data, json=json, **kwargs)
def _put(url, data=None, **kwargs): return request('put', url, data=data, **kwargs)
def _patch(url, data=None, **kwargs): return request('patch', url, data=data, **kwargs)
def _delete(url, **kwargs): return request('delete', url, **kwargs)

def _head(url, **kwargs):
    kwargs.setdefault('allow_redirects', False)
    return request('head', url, **kwargs)

requests.get, requests.options, requests.post = _get, _options, _post
requests.put, requests.patch, requests.delete, requests.head = _put, _patch, _delete, _head


def patch_httpx_func(httpx, fname):
    _old_func = getattr(httpx, fname)

    def new_func(url, **kwargs):
        if os.environ.get('http_proxy') and _is_ip_address_url(url):
            try:
                return _old_func(url, **{**kwargs, **dict(trust_env=False)})
            except Exception: pass
        return _old_func(url, **kwargs)

    setattr(httpx, fname, new_func)


def patch_httpx():
    import httpx
    sig = inspect.signature(_old_httpx_func := httpx.request)
    proxy_name = 'proxy' if 'proxy' in sig.parameters else 'proxies'

    def new_httpx_func(method, url, **kwargs):
        if (proxies := kwargs.pop('proxies', kwargs.pop('proxy', None))):
            kwargs[proxy_name] = proxies
        if os.environ.get('http_proxy') and _is_ip_address_url(url):
            try:
                return _old_httpx_func(method, url, **{**kwargs, **dict(trust_env=False)})
            except Exception: pass
        return _old_httpx_func(method, url, **kwargs)

    httpx.request = new_httpx_func

    for fname in ['get', 'options', 'post', 'delete', 'put', 'patch', 'head']:
        patch_httpx_func(httpx, fname)


class LazyPatchLoader(importlib.abc.Loader):
    """延迟补丁加载器，用于在模块加载时自动应用补丁。

``LazyPatchLoader`` 是一个导入系统加载器，它在模块执行时自动为请求库和httpx库应用补丁。
这个加载器包装了原始的模块规范，在模块执行完成后自动调用补丁函数。

Args:
    original_spec (ModuleSpec): 原始模块的规范对象，包含模块的加载信息和路径。

功能:

- 在模块加载时自动设置正确的包和路径属性
- 执行原始加载器的模块执行逻辑
- 在模块执行完成后自动应用requests和httpx库的补丁

"""
    PATCHS = {
        'httpx': patch_httpx,
    }
    PATCHED = set()

    def __init__(self, original_spec, package_name):
        self.original_spec = original_spec
        self._package_name = package_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        """执行模块的加载和初始化过程。

此方法是导入系统加载器的核心方法，负责执行模块的代码并初始化模块对象。
在LazyPatchLoader中，此方法会先设置模块的包和路径属性，然后执行原始加载器的模块执行逻辑，
最后自动为requests和httpx库应用补丁。

Args:
    module (ModuleType): 要执行的模块对象。

"""
        if self.original_spec.submodule_search_locations is not None:
            module.__package__ = self.original_spec.name
        elif '.' in self.original_spec.name:
            module.__package__ = self.original_spec.name.rpartition('.')[0]
        else:
            module.__package__ = ''

        if self.original_spec.submodule_search_locations is not None:
            module.__path__ = self.original_spec.submodule_search_locations

        self.original_spec.loader.exec_module(module)
        LazyPatchLoader.PATCHS[self._package_name]()
        LazyPatchLoader.PATCHED.add(self._package_name)

class LazyPatchFinder(importlib.abc.MetaPathFinder):
    """延迟补丁查找器，用于在导入时拦截特定模块并应用补丁。

``LazyPatchFinder`` 是一个元路径查找器，它在导入过程中拦截对'requests'和'httpx'模块的导入请求，
并使用自定义的LazyPatchLoader来加载这些模块，从而在模块加载时自动应用补丁。

**Note:**

- 此查找器只在模块尚未导入时生效
- 如果模块已经导入，会直接调用patch_requests_and_httpx()函数
- 支持模块的原始属性和路径保持完整
"""
    def find_spec(self, fullname, path, target=None):
        """查找并返回模块的规范对象，用于自定义模块加载过程。

此方法是MetaPathFinder的核心方法，负责在导入过程中查找指定模块的规范对象。
在LazyPatchFinder中，它专门拦截'requests'和'httpx'模块的导入请求，使用自定义的LazyPatchLoader来包装原始模块规范。

Args:
    fullname (str): 要导入的完整模块名称
    path (list): 搜索路径列表，对于顶级模块为None
    target (module, optional): 目标模块对象（重载时使用）

**Returns:**

- 对于'requests'和'httpx'模块：返回使用LazyPatchLoader包装的模块规范
- 对于其他模块：返回None，让其他查找器继续处理

"""
        if fullname in LazyPatchLoader.PATCHS and fullname not in LazyPatchLoader.PATCHED:
            if self in sys.meta_path: sys.meta_path.remove(self)
            original_spec = importlib.util.find_spec(fullname)
            if len(LazyPatchLoader.PATCHS) > len(LazyPatchLoader.PATCHED) + 1:
                sys.meta_path.insert(0, self)
            if original_spec is None: return None
            return importlib.util.spec_from_loader(fullname, LazyPatchLoader(original_spec, fullname),
                                                   origin=original_spec.origin)
        return None

for name, fn in LazyPatchLoader.PATCHS.items():
    if name in sys.modules:
        fn()
        LazyPatchLoader.PATCHED.add(name)

if len(LazyPatchLoader.PATCHS) != len(LazyPatchLoader.PATCHED):
    sys.meta_path.insert(0, LazyPatchFinder())


def patch_os_env(set_action: Callable[[str, str], None], unset_action: Callable[[str], None]):

    old_setitem = os._Environ.__setitem__

    def new_setitem(self, key, value):
        old_setitem(self, key, value)
        if isinstance(key, bytes): key = key.decode('utf-8')
        set_action(key, value)

    old_delitem = os._Environ.__delitem__

    def new_delitem(self, key):
        old_delitem(self, key)
        if isinstance(key, bytes): key = key.decode('utf-8')
        unset_action(key)

    os._Environ.__setitem__ = new_setitem
    os._Environ.__delitem__ = new_delitem
