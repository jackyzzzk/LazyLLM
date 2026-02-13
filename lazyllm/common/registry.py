import builtins
import functools
import lazyllm
import re
import sys
from typing import Union, List
from .bind import _MetaBind
from ..configs import config
from typing import Optional
from abc import ABCMeta

# Special Dict for lazy programmer. Suppose we have a LazyDict as follows：
#    >>> ld = LazyDict(name='ld', ALd=int)
# 1. Use dot instead of ['str']
#    >>> ld.ALd
# 2. Support lowercase first character to make the sentence more like a function
#    >>> ld.aLd
# 3. Supports direct calls to dict when there is only one element
#    >>> ld()
# 4. Support dynamic default key
#    >>> ld.set_default('ALd')
#    >>> ld.default
# 5. allowed to omit the group name if the group name appears in the name
#    >>> ld.a
class LazyDict(dict):
    """一个为懒惰的程序员设计的特殊字典类。支持多种便捷的访问和操作方式。

特性：

1. 使用点号代替['str']访问字典元素 
2. 支持首字母小写来使语句更像函数调用
3. 当字典只有一个元素时支持直接调用
4. 支持动态默认键
5. 如果组名出现在名称中，允许省略组名

Args:
    name (str): 字典的名称，默认为空字符串。
    base: 基类引用，默认为None。
    *args: 位置参数，传递给dict父类。
    **kw: 关键字参数，传递给dict父类。
"""
    def __init__(self, name='', base=None, *args, **kw):
        super(__class__, self).__init__(*args, **kw)
        self._default: Optional[str] = None
        self.name = name.capitalize()
        self.base = base

    def __setitem__(self, key, value):
        key = key.lower()
        assert key != 'default', 'LazyDict do not support key: default'
        if '.' in key:
            grp, key = key.rsplit('.', 1)
            return self[grp].__setitem__(key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        key = key.lower()
        if '.' in key:
            grp, key = key.split('.', 1)
            return self[grp][key]
        return super().__getitem__(key)

    # default -> self.default
    # key -> Key, keyName, KeyName
    # if self.name ends with 's' or 'es', ignor it
    def _match(self, key: str):
        key = key.lower()
        if key == 'default':
            assert self._default or len(self) > 0, 'No default key set'
            key = self._default or list(self.keys())[0]
        keys = [key, f'{key}{self.name}', f'{key}{self.name.lower()}']
        if self.name.endswith('s'):
            n = 2 if self.name.endswith('es') else 1
            keys.extend([f'{key}{self.name[:-n]}', f'{key}{self.name[:-n].lower()}'])

        for k in set(keys):
            if k in self.keys():
                return k
        raise AttributeError(f'Attr {key} not found in `{self.name}: {self}`, conditates: {keys}')

    def __getattr__(self, key):
        return self[self._match(key)]

    def remove(self, key):
        """从字典中移除指定的键值对。

Args:
    key (str): 要移除的键。支持与__getattr__相同的键匹配规则，包括首字母小写和组名省略等特性。

注意:
    如果找不到匹配的键，将抛出AttributeError异常。
"""
        super(__class__, self).pop(self._match(key))

    def __call__(self, *args, **kwargs):
        assert self._default is not None or len(self.keys()) == 1
        return (self.default if self._default else self[list(self.keys())[0]])(*args, **kwargs)

    def set_default(self, key: str):
        """设置字典的默认键。设置后可以通过.default属性访问该键对应的值。

Args:
    key (str): 要设置为默认的键名。

注意:
    - key必须是字符串类型
    - 设置后可以通过.default访问，或在字典只有一个元素时直接调用
"""
        assert isinstance(key, str), 'default key must be str'
        self._default = key.lower()

    def __contains__(self, key):
        try:
            _ = self[self._match(key)]
            return True
        except (AttributeError, KeyError):
            return False


group_template = '''\
class LazyLLM{name}Base(LazyLLMRegisterMetaClass.all_clses[\'{base}\'.lower()].base):
    pass
'''

config.add('use_builtin', bool, False, 'USE_BUILTIN',
           description='Whether to use registry modules in python builtin.')


class LazyLLMRegisterMetaClass(_MetaBind):
    all_clses = LazyDict()

    def __new__(metas, name, bases, attrs):
        new_cls = type.__new__(metas, name, bases, attrs)
        if attrs.get('__lazyllm_registry_disable__', False) is True: return new_cls
        if name.startswith('LazyLLM') and name.endswith('Base'):
            ori = new_cls.__dict__.get('__lazyllm_registry_key__', re.match('(LazyLLM)(.*)(Base)', name)[2])
            group = ori.lower()
            ori_group = getattr(new_cls, '_lazy_llm_group', '')
            new_cls._lazy_llm_group = f'{ori_group}.{group}'.strip('.')
            ld = LazyDict(group, new_cls)
            if new_cls._lazy_llm_group == group:
                for m in (builtins, lazyllm) if config['use_builtin'] else (lazyllm,):
                    assert not (hasattr(m, group) and hasattr(m, ori)), f'group name \'{ori}\' cannot be used'
                for m in (builtins, lazyllm) if config['use_builtin'] else (lazyllm,):
                    setattr(m, group, ld)
                    setattr(m, ori, ld)
            LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group] = ld
            if (f := getattr(new_cls, '__lazyllm_after_registry_hook__', None)):
                f(new_cls, ori_group, group, isleaf=False)
        elif hasattr(new_cls, '_lazy_llm_group'):
            group = LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group]
            name = new_cls.__dict__.get('__lazyllm_registry_key__', name)
            assert name not in group, f'duplicate class \'{name}\' in group {new_cls._lazy_llm_group}'
            group[name] = new_cls
            if (f := getattr(new_cls, '__lazyllm_after_registry_hook__', None)):
                f(new_cls, new_cls._lazy_llm_group, name, isleaf=True)
        return new_cls


class LazyLLMRegisterMetaABCClass(LazyLLMRegisterMetaClass, ABCMeta): pass


def _get_base_cls_from_registry(cls_str, *, registry=LazyLLMRegisterMetaClass.all_clses):
    if cls_str == '':
        return registry.base
    group, cls_str = cls_str.split('.', 1) if '.' in cls_str else (cls_str, '')
    if not (registry is LazyLLMRegisterMetaClass.all_clses or group in registry):
        exec(group_template.format(name=group.capitalize(), base=registry.base._lazy_llm_group))
    return _get_base_cls_from_registry(cls_str, registry=registry[group])


reg_template = '''\
class {name}(LazyLLMRegisterMetaClass.all_clses[\'{base}\'.lower()].base):
    pass
'''

def bind_to_instance(func):
    @functools.wraps(func)
    def wrapper(instance, *args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class Register(object):
    """LazyLLM提供了一套组件注册机制，允许将任意函数注册为LazyLLM的Component。通过注册器提供的分组机制，注册后的函数可在任意位置通过分组索引进行调用，无需显式导入。

<span style="font-size: 18px;">&ensp;**`lazyllm.components.register(cls, *, rewrite_func)→ 装饰器`**</span>

该函数调用后返回一个装饰器，将被装饰函数包装为Component并注册到名为cls的分组中。

Args:
    base (type): 基类
    fnames (Union[str, List[str]]): 要重写的函数名或函数名列表
    template (str, optional): 注册模板字符串，默认为标准注册模板
    default_group (str, optional): 默认组名，默认为None


Examples:
    >>> import lazyllm
    >>> @lazyllm.component_register('mygroup')
    ... def myfunc(input):
    ...    return input
    ...
    >>> lazyllm.mygroup.myfunc()(1)
    1
    >>> @lazyllm.component_register.cmd('mygroup')
    ... def mycmdfunc(input):
    ...     return f'echo {input}'
    ...
    >>> lazyllm.mygroup.mycmdfunc()(1)
    PID: 2024-06-01 00:00:00 lazyllm INFO: (lazyllm.launcher) Command: echo 1
    PID: 2024-06-01 00:00:00 lazyllm INFO: (lazyllm.launcher) PID: 1
    """
    def __init__(self, base, fnames, template: str = reg_template, default_group: Optional[str] = None,
                 allowed_parameter: Optional[Union[str, List[str]]] = None):
        self.basecls = base
        self.fnames = [fnames] if isinstance(fnames, str) else fnames
        self.template = template
        self._default_group = default_group
        if isinstance(allowed_parameter, str):
            self._allowed_parameter = {allowed_parameter}
        elif isinstance(allowed_parameter, list):
            assert all(isinstance(p, str) for p in allowed_parameter), 'allowed_parameter must be list of str'
            self._allowed_parameter = set(allowed_parameter)
        elif allowed_parameter is None:
            self._allowed_parameter = set()
        else:
            raise TypeError('allowed_parameter must be str or list[str]')
        assert len(self.fnames) > 0, 'At least one function should be given for overwrite.'

    def _wrap(self, cls, *, rewrite_func=None, **kwargs):
        cls = cls.__name__ if isinstance(cls, type) else cls
        cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2] \
            if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls
        base = _get_base_cls_from_registry(cls.lower())
        assert issubclass(base, self.basecls)
        if rewrite_func is None:
            rewrite_func = base.__reg_overwrite__ if getattr(base, '__reg_overwrite__', None) else self.fnames[0]
        assert rewrite_func in self.fnames, f'Invalid function "{rewrite_func}" provived for rewrite.'

        def impl(func, func_name=None):
            if func_name:
                func_for_wrapper = func  # avoid calling recursively

                @functools.wraps(func)
                def wrapper_func(*args, **kwargs):
                    return func_for_wrapper(*args, **kwargs)

                wrapper_func.__name__ = func_name
                func = wrapper_func
            else:
                func_name = func.__name__
            exec(self.template.format(
                name=func_name + cls.split('.')[-1].capitalize(), base=cls))
            # 'func' cannot be recognized by exec, so we use 'setattr' instead
            f = LazyLLMRegisterMetaClass.all_clses[cls.lower()].__getattr__(func_name)

            # Support multiprocessing: Register the class in the module where the function is defined
            if func.__module__ in sys.modules:
                setattr(sys.modules[func.__module__], f.__name__, f)
                f.__module__ = func.__module__

            f.__name__ = func_name
            setattr(f, rewrite_func, bind_to_instance(func))
            for k, v in kwargs.items():
                setattr(f, k, v)
            return func
        return impl

    def __call__(self, f, *, rewrite_func=None, **kwargs):
        if not isinstance(f, (str, type)):
            assert self._default_group, 'default_group is not set, please set it by your register decorator'
            return self._wrap(self._default_group)(f)
        assert all(k in self._allowed_parameter for k in kwargs.keys()), \
            f'Only allowed parameters: {self._allowed_parameter}, but got {kwargs.keys()}'
        return self._wrap(f, rewrite_func=rewrite_func, **kwargs)

    def __getattr__(self, name):
        if name not in self.fnames:
            raise AttributeError(f'class {self.__class__} has no attribute {name}')

        def impl(cls):
            return self(cls, rewrite_func=name)
        return impl

    def new_group(self, group_name):
        """创建一个新的ComponentGroup。新建的group会自动加入到__builtin__中，无需import即可在任一位置访问到该group。

Args:
    group_name (str): 待创建的group的名字
"""
        return type(f'LazyLLM{group_name}Base', (self.basecls,), {})
