import os
import re
import time
import uuid
import copy
import psutil
import random
import threading
import subprocess
import multiprocessing
from enum import Enum
from datetime import datetime
from collections import defaultdict

import lazyllm
from lazyllm.common import RecentQueue as Queue
from lazyllm import LazyLLMRegisterMetaClass, LazyLLMCMD, final, LOG


class Status(Enum):
    """An enumeration."""
    TBSubmitted = 0,
    InQueue = 1
    Running = 2,
    Pending = 3,
    Done = 100,
    Cancelled = 101,  # TODO(wangzhihong): support cancel job
    Failed = 102,


class LazyLLMLaunchersBase(object, metaclass=LazyLLMRegisterMetaClass):
    """用于统一管理外部进程或分布式作业（训练/推理等）生命周期的启动器抽象基类。不同平台（本地、SLURM、K8s、云资源等）的具体启动器应继承该类并实现核心接口。

Args:
    None.
"""
    Status = Status

    def __init__(self) -> None:
        self._id = str(uuid.uuid4().hex)

    def makejob(self, cmd):
        """根据给定命令创建并返回作业/进程句柄。需由子类实现。

Args:
    cmd: 用于创建作业的命令或配置（如字符串、参数列表或作业描述对象）。

Raises:
    NotImplementedError: 基类未实现，子类必须覆盖。
"""
        raise NotImplementedError

    def launch(self, *args, **kw):
        """启动一个或多个作业，并将其登记到 all_processes[self._id] 中。需由子类实现。

Args:
    *args: 与具体实现相关的位置参数。
    **kw: 与具体实现相关的关键字参数。

Raises:
    NotImplementedError: 基类未实现，子类必须覆盖。
"""
        raise NotImplementedError

    def cleanup(self):
        """停止并清理当前启动器登记的所有作业，从 all_processes 中移除相应记录，并在最后阻塞等待作业结束。

Args:
    None.
"""
        for k, v in self.all_processes[self._id]:
            v.stop()
            LOG.info(f'killed job:{k}')
        self.all_processes.pop(self._id)
        self.wait()

    @property
    def status(self):
        if len(self.all_processes[self._id]) == 1:
            return self.all_processes[self._id][0][1].status
        elif len(self.all_processes[self._id]) == 0:
            return Status.Cancelled
        raise RuntimeError('More than one tasks are found in one launcher!')

    @property
    def log_path(self):
        if len(self.all_processes[self._id]) == 1:
            return self.all_processes[self._id][0][1].log_path
        elif len(self.all_processes[self._id]) == 0:
            return None
        raise RuntimeError('More than one tasks are found in one launcher!')

    def wait(self):
        """阻塞等待当前启动器登记的所有作业结束。

Args:
    None.
"""
        for _, v in self.all_processes[self._id]:
            v.wait()

    def clone(self):
        """深拷贝当前启动器实例并分配新的唯一 _id，返回克隆后的实例。

Args:
    None.

**Returns:**

- LazyLLMLaunchersBase: 克隆出的启动器实例。
"""
        new = copy.deepcopy(self)
        new._id = str(uuid.uuid4().hex)
        return new


lazyllm.launchers['Status'] = Status

lazyllm.config.add('launcher', str, 'empty', 'DEFAULT_LAUNCHER',
                   description='The default remote launcher to use if no launcher is specified.')
lazyllm.config.add('cuda_visible', bool, False, 'CUDA_VISIBLE',
                   description='Whether to set the CUDA_VISIBLE_DEVICES environment variable.')


# store cmd, return message and command output.
# LazyLLMCMD's post_function can get message form this class.
class Job(object):
    """通用任务调度执行类。
该类用于封装一个通过启动器（launcher）调度执行的任务，支持命令包装、同步控制、返回值提取、命令固定等功能。

Args:
    cmd (LazyLLMCMD): 要执行的命令对象。
    launcher (Any): 启动器实例，用于实际任务调度执行。
    sync (bool): 是否为同步执行，默认为 True。
"""
    def __init__(self, cmd, launcher, *, sync=True):
        assert isinstance(cmd, LazyLLMCMD)
        self._origin_cmd = cmd
        self.sync = sync
        self._launcher = launcher
        self.queue, self.jobid, self.ip, self.ps = Queue(), None, None, None
        self.output_hooks = []

    def _set_return_value(self):
        cmd = getattr(self, '_fixed_cmd', None)
        if cmd and callable(cmd.return_value):
            self.return_value = cmd.return_value(self)
        elif cmd and cmd.return_value:
            self.return_value = cmd.return_value
        else:
            self.return_value = self

    def get_executable_cmd(self, *, fixed=False):
        """生成最终可执行命令。
如果已缓存固定命令（fixed），则直接返回。否则根据原始命令进行包裹（wrap）并缓存为 `_fixed_cmd`。

Args:
    fixed (bool): 是否使用已固定的命令对象（若已存在）。

**Returns:**

- LazyLLMCMD: 可直接执行的命令对象。
"""
        if fixed and hasattr(self, '_fixed_cmd'):
            LOG.info('Command is fixed!')
            return self._fixed_cmd
        cmd = self._origin_cmd
        if callable(cmd.cmd):
            cmd = cmd.with_cmd(cmd.cmd())
        self._fixed_cmd = cmd.with_cmd(self._wrap_cmd(cmd.cmd))
        return self._fixed_cmd

    # interfaces
    def stop(self):
        """停止当前作业。
该方法为接口定义，需子类实现，当前抛出 NotImplementedError。
"""
        raise NotImplementedError

    @property
    def status(self):
        """当前作业状态。
该属性为接口定义，需子类实现，当前抛出 NotImplementedError。
"""
        raise NotImplementedError

    def wait(self):
        """挂起当前线程，等待作业执行完成。当前实现为空方法（子类可重写）。
"""
        pass

    def _wrap_cmd(self, cmd): return cmd

    def _start(self, *, fixed):
        cmd = self.get_executable_cmd(fixed=fixed)
        LOG.info(f'Command: {cmd}')
        if lazyllm.config['mode'] == lazyllm.Mode.Display: return
        self.ps = subprocess.Popen(cmd.cmd, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        self._get_jobid()
        self._enqueue_subprocess_output(hooks=self.output_hooks)

        if self.sync:
            self.ps.wait()
        else:
            self._launcher.all_processes[self._launcher._id].append((self.jobid, self))
            n = 0
            while self.status in (Status.TBSubmitted, Status.InQueue, Status.Pending):
                time.sleep(2)
                n += 1
                if n > 1800:  # 3600s
                    self._launcher.all_processes[self._launcher._id].pop()
                    LOG.error('Launch failed: No computing resources are available.')
                    break

    def restart(self, *, fixed=False):
        """重新启动作业流程。
该函数会先停止已有进程，等待 2 秒后重新启动作业。

Args:
    fixed (bool): 是否使用固定后的命令。
"""
        self.stop()
        time.sleep(2)
        self._start(fixed=fixed)

    def start(self, *, restart=3, fixed=False):
        """对外接口：启动作业，并支持失败时的自动重试。
若作业执行失败，会根据 `restart` 参数控制重试次数。

Args:
    restart (int): 重试次数。默认为 3。
    fixed (bool): 是否使用固定后的命令。用于避免多次构建。
"""
        self._start(fixed=fixed)
        if not (lazyllm.config['mode'] == lazyllm.Mode.Display or self._fixed_cmd.checkf(self)):
            if restart > 0:
                for ii in range(restart):
                    LOG.warning(f'Job failed, restarting... ({ii + 1}/{restart})')
                    self.restart(fixed=fixed)
                    if self._fixed_cmd.checkf(self): break
                else:
                    detail = self.queue.get_recent(join='\n', join_prefix=' Last logs:\n')
                    raise RuntimeError(f'Job failed after retrying {restart} times.{detail}')
            else:
                detail = self.queue.get_recent(join='\n', join_prefix=' Last logs:\n')
                raise RuntimeError(f'Job failed without retries.{detail}')
        self._set_return_value()

    def _enqueue_subprocess_output(self, hooks=None):
        self.output_thread_event = threading.Event()

        def impl(out, queue):
            for line in iter(out.readline, b''):
                try:
                    line = line.decode('utf-8')
                except Exception:
                    try:
                        line = line.decode('gb2312')
                    except Exception:
                        pass
                if isinstance(line, str):
                    queue.put(line)
                    if hooks:
                        hooks(line) if callable(hooks) else [hook(line) for hook in hooks]
                LOG.info(f'{line.lstrip("INFO:").rstrip()}', jobid=self.jobid, name='launcher')
                if self.output_thread_event.is_set():
                    break
            out.close()
        self.output_thread = threading.Thread(target=impl, args=(self.ps.stdout, self.queue))
        self.output_thread.daemon = True
        self.output_thread.start()

    def _generate_name(self):
        now = datetime.now()
        return str(hex(hash(now.strftime('%S%M') + str(random.randint(3, 2000)))))[2:10]

    def __deepcopy__(self, memo=None):
        raise RuntimeError('Cannot copy Job object')

    @property
    def log_path(self):
        match = re.search(r'tee\s+([^\s]+\.log)', self._origin_cmd.cmd)
        if match:
            return match.group(1)
        return None


@final
class EmptyLauncher(LazyLLMLaunchersBase):
    """此类是 ``LazyLLMLaunchersBase`` 的子类，作为一个本地的启动器。

Args:
    subprocess (bool): 是否使用子进程来启动。默认为 `False`。
    sync (bool): 是否同步执行作业。默认为 `True`，否则为异步执行。


Examples:
    >>> import lazyllm
    >>> launcher = lazyllm.launchers.empty()
    """
    all_processes = defaultdict(list)

    @final
    class Job(Job):
        """通用任务调度执行类。
该类用于封装一个通过启动器（launcher）调度执行的任务，支持命令包装、同步控制、返回值提取、命令固定等功能。

Args:
    cmd (LazyLLMCMD): 要执行的命令对象。
    launcher (Any): 启动器实例，用于实际任务调度执行。
    sync (bool): 是否为同步执行，默认为 True。
"""
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, launcher, sync=sync)

        def _wrap_cmd(self, cmd):
            if self._launcher.ngpus == 0:
                return cmd
            gpus = self._launcher._get_idle_gpus()
            if gpus and lazyllm.config['cuda_visible']:
                if self._launcher.ngpus is None:
                    empty_cmd = f'export CUDA_VISIBLE_DEVICES={gpus[0]} && '
                elif self._launcher.ngpus <= len(gpus):
                    empty_cmd = 'export CUDA_VISIBLE_DEVICES=' + \
                                ','.join([str(n) for n in gpus[:self._launcher.ngpus]]) + ' && '
                else:
                    error_info = (f'Not enough GPUs available. Requested {self._launcher.ngpus} GPUs, '
                                  f'but only {len(gpus)} are available.')
                    LOG.error(error_info)
                    raise error_info
            else:
                empty_cmd = ''
            return empty_cmd + cmd

        def stop(self):
            if self.ps:
                try:
                    parent = psutil.Process(self.ps.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    LOG.warning(f'Process with PID {self.ps.pid} does not exist.')
                except psutil.AccessDenied:
                    LOG.warning(f'Permission denied when trying to kill process with PID {self.ps.pid}.')
                except Exception as e:
                    LOG.warning(f'An error occurred: {e}')

        @property
        def status(self):
            return_code = self.ps.poll()
            if return_code is None: job_status = Status.Running
            elif return_code == 0: job_status = Status.Done
            else: job_status = Status.Failed
            return job_status

        def _get_jobid(self):
            self.jobid = self.ps.pid if self.ps else None

        def get_jobip(self):
            return '127.0.0.1'

        def wait(self):
            if self.ps:
                self.ps.wait()

    def __init__(self, subprocess=False, ngpus=None, sync=True, **kwargs):
        super().__init__()
        self.subprocess = subprocess
        self.sync = sync
        self.ngpus = ngpus

    def makejob(self, cmd):
        return EmptyLauncher.Job(cmd, launcher=self, sync=self.sync)

    def launch(self, f, *args, **kw):
        if isinstance(f, EmptyLauncher.Job):
            f.start()
            return f.return_value
        elif callable(f):
            if not self.subprocess:
                return f(*args, **kw)
            else:
                LOG.info('Async execution of callable object is not supported currently.')
                p = multiprocessing.Process(target=f, args=args, kwargs=kw)
                p.start()
                p.join()
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')

    def _get_idle_gpus(self):
        try:
            order_list = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
        except Exception as e:
            LOG.warning(f'Get idle gpus failed: {e}, if you have no gpu-driver, ignor it.')
            return []
        lines = order_list.strip().split('\n')

        str_num = os.getenv('CUDA_VISIBLE_DEVICES', None)
        if str_num:
            sub_gpus = [int(x) for x in str_num.strip().split(',')]

        gpu_info = []
        for line in lines:
            index, memory_free = line.split(', ')
            if not str_num or int(index) in sub_gpus:
                gpu_info.append((int(index), int(memory_free)))
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        LOG.info('Memory left:\n' + '\n'.join([f'{item[0]} GPU, left: {item[1]} MiB' for item in gpu_info]))
        return [info[0] for info in gpu_info]

class RemoteLauncher(LazyLLMLaunchersBase):
    """此类是 ``LazyLLMLaunchersBase`` 的一个子类，它充当了一个远程启动器的代理。它根据配置文件中的 ``lazyllm.config['launcher']`` 条目动态地创建并返回一个对应的启动器实例(例如：``SlurmLauncher`` 或 ``ScoLauncher``)。

Args:
    *args: 位置参数，将传递给动态创建的启动器构造函数。
    sync (bool): 是否同步执行作业。默认为 ``False``。
    **kwargs: 关键字参数，将传递给动态创建的启动器构造函数。

注意事项: 
    - ``RemoteLauncher`` 不是一个直接的启动器，而是根据配置动态创建一个启动器。 
    - 配置文件中的 ``lazyllm.config['launcher']`` 指定一个存在于 ``lazyllm.launchers`` 模块中的启动器类名。该配置可通过设置环境变量 ``LAZYLLM_DEFAULT_LAUNCHER`` 来设置。如：``export LAZYLLM_DEFAULT_LAUNCHER=sco`` , ``export LAZYLLM_DEFAULT_LAUNCHER=slurm`` 。


Examples:
    >>> import lazyllm
    >>> launcher = lazyllm.launchers.remote(ngpus=1)
    """
    def __new__(cls, *args, sync=False, ngpus=1, **kwargs):
        return getattr(lazyllm.launchers, lazyllm.config['launcher'])(*args, sync=sync, ngpus=ngpus, **kwargs)
