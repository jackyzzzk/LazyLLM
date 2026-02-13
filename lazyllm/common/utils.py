from os import PathLike, makedirs
from os.path import expanduser, expandvars, isfile, join, normpath
from typing import Union, Dict, Callable, Any, Optional
import re
import os
from contextlib import contextmanager
import cloudpickle
import ast
import pickle
import base64
import argparse

def check_path(
    path: Union[str, PathLike],
    exist: bool = True,
    file: bool = True,
    parents: bool = True,
) -> str:
    # normalize and expand a path
    path = normpath(expandvars(expanduser(path)))
    if exist and file and not isfile(path):
        raise FileNotFoundError(path)
    else:
        if file:
            dir_path = normpath(join(path, '..'))
        else:
            dir_path = path
        if parents:
            makedirs(dir_path, exist_ok=True)
    return path

class SecurityVisitor(ast.NodeVisitor):  # noqa C901
    """基于AST的Python代码安全分析器，用于检测不安全的操作。

属性：
    DANGEROUS_BUILTINS (set): 危险的内置函数集合，包括exec、eval、open等。
    DANGEROUS_OS_CALLS (set): 危险的os操作集合，包括system、popen、remove等。
    DANGEROUS_SYS_CALLS (set): 危险的sys操作集合，包括exit、modules等。
    DANGEROUS_MODULES (set): 危险的模块集合，包括pickle、subprocess、socket等。

注意：
    此类继承自ast.NodeVisitor，用于遍历和检查Python代码的抽象语法树。
"""

    # **Dangerous built-in functions**
    DANGEROUS_BUILTINS = {'exec', 'eval', 'open', 'compile', 'getattr',
                          'setattr', '__import__', 'globals', 'locals', 'vars'}

    # **Dangerous os operations**
    DANGEROUS_OS_CALLS = {'system', 'popen', 'remove', 'rmdir', 'unlink', 'rename'}

    # **Dangerous sys operations**
    DANGEROUS_SYS_CALLS = {'exit', 'modules'}

    # **Dangerous modules**
    DANGEROUS_MODULES = {'pickle', 'subprocess', 'socket', 'shutil', 'requests', 'inspect', 'tempfile'}

    def visit_Call(self, node):  # noqa C901
        """检查函数调用的安全性。

检查内容：
1. 危险的内置函数调用
2. 危险的os模块调用
3. 危险的sys模块调用

Args:
    node (ast.Call): AST函数调用节点。

Raises:
    ValueError: 当检测到危险的函数调用时抛出。
"""
        # Direct calls to dangerous built-in functions
        if isinstance(node.func, ast.Name) and node.func.id in self.DANGEROUS_BUILTINS:
            raise ValueError(f'⚠️ Detected dangerous function call: {node.func.id}')

        # Check for __import__ calls with string arguments
        if isinstance(node.func, ast.Name) and node.func.id == '__import__':
            if node.args and isinstance(node.args[0], ast.Str):
                module_name = node.args[0].s
                if module_name in self.DANGEROUS_MODULES:
                    raise ValueError(f'⚠️ Detected dangerous module import via __import__: {module_name}')

        # Check for indirect __import__ calls (function calls that might return __import__)
        if isinstance(node.func, ast.Call):
            # Check if this is a call to a function that might return __import__
            if isinstance(node.func.func, ast.Name):
                func_name = node.func.func.id
                if func_name in ['get_import', 'import_func']:  # Common patterns
                    raise ValueError(f'⚠️ Detected suspicious function call that might return __import__: {func_name}')

        # Check for attribute access that might lead to __import__
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'ImportHelper':
                if node.func.attr == 'get_import':
                    raise ValueError('⚠️ Detected suspicious method call: ImportHelper.get_import')

        # os / sys related calls
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'os' and node.func.attr in self.DANGEROUS_OS_CALLS:
                raise ValueError(f'⚠️ Detected dangerous os call: os.{node.func.attr}')
            if node.func.value.id == 'sys' and node.func.attr in self.DANGEROUS_SYS_CALLS:
                raise ValueError(f'⚠️ Detected dangerous sys call: sys.{node.func.attr}')

        self.generic_visit(node)

    def visit_Import(self, node):
        """检查import语句的安全性。

Args:
    node (ast.Import): AST导入节点。

Raises:
    ValueError: 当检测到危险模块的导入时抛出。
"""
        for alias in node.names:
            if alias.name in self.DANGEROUS_MODULES:
                raise ValueError(f'⚠️ Detected dangerous module import: {alias.name}')

    def visit_ImportFrom(self, node):
        """检查from...import语句的安全性。

Args:
    node (ast.ImportFrom): AST from-import节点。

Raises:
    ValueError: 当检测到危险模块的导入时抛出。
"""
        if node.module in self.DANGEROUS_MODULES:
            raise ValueError(f'⚠️ Detected dangerous module import: {node.module}')

    def visit_Attribute(self, node):
        """检查属性访问的安全性。

检查内容：
1. os.environ的访问
2. tempfile模块的使用

Args:
    node (ast.Attribute): AST属性访问节点。

Raises:
    ValueError: 当检测到危险的属性访问时抛出。
"""
        if isinstance(node.value, ast.Name):
            if node.value.id == 'os' and node.attr == 'environ':
                raise ValueError('⚠️ Detected dangerous access: os.environ')
            if node.value.id == 'tempfile':
                raise ValueError(f'⚠️ Detected dangerous usage of tempfile: tempfile.{node.attr}')

        self.generic_visit(node)

    def visit_Lambda(self, node):
        # Check if lambda body returns __import__
        if isinstance(node.body, ast.Name) and node.body.id == '__import__':
            raise ValueError('⚠️ Detected lambda function returning __import__')
        self.generic_visit(node)

    def visit_ListComp(self, node):
        # Check if the expression in list comprehension is __import__
        if isinstance(node.elt, ast.Name) and node.elt.id == '__import__':
            raise ValueError('⚠️ Detected list comprehension containing __import__')
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Check if function returns __import__
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name) and stmt.value.id == '__import__':
                    raise ValueError(f'⚠️ Detected function {node.name} returning __import__')
        self.generic_visit(node)

def compile_func(func_code: str, global_env: Optional[Dict[str, Any]] = None) -> Callable:
    """
将一段 python 函数字符串编译成一个可执行函数并返回。

Args:
    func_code (str): 包含 python 函数代码的字符串
    global_env (str): 在 python 函数中用到的包和全局变量


Examples:

    from lazyllm.common import compile_func
    code_str = 'def Identity(v): return v'
    identity = compile_func(code_str)
    assert identity('hello') == 'hello'
    """
    fname = re.search(r'def\s+(\w+)\s*\(', func_code).group(1)
    module = ast.parse(func_code)
    SecurityVisitor().visit(module)
    func = compile(module, filename='<ast>', mode='exec')
    local_dict = {}
    exec(func, global_env if global_env is not None else local_dict, local_dict)
    return local_dict.pop(fname)

def obj2str(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def str2obj(data: str) -> Any:
    return None if data is None else pickle.loads(base64.b64decode(data.encode('utf-8')))

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dump_obj(f):
    @contextmanager
    def env_helper():
        os.environ['LAZYLLM_ON_CLOUDPICKLE'] = 'ON'
        yield
        os.environ['LAZYLLM_ON_CLOUDPICKLE'] = 'OFF'

    with env_helper():
        return None if f is None else base64.b64encode(cloudpickle.dumps(f)).decode('utf-8')

def load_obj(f):
    return cloudpickle.loads(base64.b64decode(f.encode('utf-8')))
