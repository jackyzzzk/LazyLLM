import re
import time
import subprocess
from collections import defaultdict

import lazyllm
from lazyllm import final
from lazyllm.common import RecentQueue as Queue
from .base import LazyLLMLaunchersBase, Job, Status

lazyllm.config.add('partition', str, 'your_part', 'SLURM_PART',
                   description='The default slurm partition to use if no partition is specified.')


@final
class SlurmLauncher(LazyLLMLaunchersBase):
    """此类是 ``LazyLLMLaunchersBase`` 的子类，作为 Slurm 启动器。

具体而言，它提供了启动和配置 Slurm 作业的方法，包括指定分区、节点数量、进程数量、GPU 数量以及超时时间等参数。

Args:
    partition (str): 要使用的 Slurm 分区。默认为 ``None``，此时将使用 ``lazyllm.config['partition']`` 中的默认分区。
                     该配置可通过设置环境变量来生效，如 ``export LAZYLLM_SLURM_PART=a100``。
    nnode (int): 要使用的节点数量。默认为 ``1``。
    nproc (int): 每个节点要使用的进程数量。默认为 ``1``。
    ngpus (int): 每个节点要使用的 GPU 数量。默认为 ``None``，即不使用 GPU。
    timeout (int): 作业的超时时间（以秒为单位）。默认为 ``None``，此时将不设置超时时间。
    sync (bool): 是否同步执行作业。默认为 ``True``，否则为异步执行。
    **kwargs: 额外参数，其中支持：
        - num_can_use_nodes (int): 可使用的最大节点数。默认为 ``5``。


Examples:
    >>> import lazyllm
    >>> launcher = lazyllm.launchers.slurm(partition='partition_name', nnode=1, nproc=1, ngpus=1, sync=False)
    """
    # In order to obtain the jobid to monitor and terminate the job more
    # conveniently, only one srun command is allowed in one Job
    all_processes = defaultdict(list)
    count = 0

    @final
    class Job(Job):
        """通用任务调度执行类。
该类用于封装一个通过启动器（launcher）调度执行的任务，支持命令包装、同步控制、返回值提取、命令固定等功能。

Args:
    cmd (LazyLLMCMD): 要执行的命令对象。
    launcher (Any): 启动器实例，用于实际任务调度执行。
    sync (bool): 是否为同步执行，默认为 True。
"""
        def __init__(self, cmd, launcher, *, sync=True, **kw):
            super(__class__, self).__init__(cmd, launcher, sync=sync)
            self.name = self._generate_name()

        def _wrap_cmd(self, cmd):
            # Assemble the order
            slurm_cmd = f'srun -p {self._launcher.partition} -N {self._launcher.nnode} --job-name={self.name}'
            if self._launcher.nproc:
                slurm_cmd += f' -n{self._launcher.nproc}'
            if self._launcher.timeout:
                slurm_cmd += f' -t {self._launcher.timeout}'
            if self._launcher.ngpus:
                slurm_cmd += f' --gres=gpu:{self._launcher.ngpus}'
            return f'{slurm_cmd} bash -c "{cmd}"'

        def _get_jobid(self):
            time.sleep(0.5)  # Wait for cmd to be stably submitted to slurm
            id_str = subprocess.check_output(['squeue', '--name=' + self.name, '--noheader'])
            if id_str:
                id_list = id_str.decode().strip().split()
                self.jobid = id_list[0]

        def get_jobip(self):
            id_str = subprocess.check_output(['squeue', '--name=' + self.name, '--noheader'])
            id_list = id_str.decode().strip().split()
            self.ip = id_list[10]
            return self.ip

        def stop(self):
            if self.jobid:
                cmd = f'scancel --quiet {self.jobid}'
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 encoding='utf-8', executable='/bin/bash')
                self.jobid = None

            if self.ps:
                self.ps.terminate()
                self.queue = Queue()
                self.output_thread_event.set()
                self.output_thread.join()

        def wait(self):
            if self.ps:
                self.ps.wait()

        @property
        def status(self):
            # lookup job
            if self.jobid:
                jobinfo = subprocess.check_output(['scontrol', 'show', 'job', str(self.jobid)])
                job_state = None
                job_state = None
                for line in jobinfo.decode().split('\n'):
                    if 'JobState' in line:
                        job_state = line.strip().split()[0].split('=')[1].strip().lower()
                        if job_state == 'running':
                            return Status.Running
                        elif job_state == 'tbsubmitted':
                            return Status.TBSubmitted
                        elif job_state == 'inqueue':
                            return Status.InQueue
                        elif job_state == 'pending':
                            return Status.Pending
                        elif job_state == 'done':
                            return Status.Done
                        elif job_state == 'cancelled':
                            return Status.Cancelled
                        else:
                            return Status.Failed
            else:
                return Status.Failed

    # TODO(wangzhihong): support configs; None -> lookup config
    def __init__(self, partition=None, nnode=1, nproc=1, ngpus=None, timeout=None, *, sync=True, **kwargs):
        super(__class__, self).__init__()
        # TODO: global config
        self.partition = partition if partition else lazyllm.config['partition']
        self.nnode, self.nproc, self.ngpus, self.timeout = nnode, nproc, ngpus, timeout
        self.sync = sync
        self.num_can_use_nodes = kwargs.get('num_can_use_nodes', 5)

    def makejob(self, cmd):
        """创建并返回一个 SlurmLauncher.Job 对象。

Args:
    cmd: 要执行的命令字符串。

**Returns:**

- SlurmLauncher.Job: 配置好的 Slurm 作业对象。
"""
        return SlurmLauncher.Job(cmd, launcher=self, sync=self.sync)

    def _add_dict(self, node_ip, used_gpus, node_dict):
        if node_ip not in node_dict:
            node_dict[node_ip] = 8 - used_gpus
        else:
            node_dict[node_ip] -= used_gpus

    def _expand_nodelist(self, nodes_str):
        pattern = r'\[(.*?)\]'
        matches = re.search(pattern, nodes_str)
        result = []
        if matches:
            nums = matches.group(1).split(',')
            base = nodes_str.split('[')[0]
            result = [base + str(x) for x in nums]
        return result

    def get_idle_nodes(self, partion=None):
        """获取指定分区中当前可用的节点数量，基于可用 GPU 数量。

该方法通过查询 Slurm 队列状态和节点信息，计算每个节点的可用 GPU 数量，并返回一个字典，其中键为节点 IP，值为可用 GPU 数量。

Args:
    partion (str, optional): 要查询的分区名称。默认为 ``None``，此时使用当前启动器的分区。

**Returns:**

- dict: 以节点 IP 为键、可用 GPU 数量为值的字典。
"""
        if not partion:
            partion = self.partition
        num_can_use_nodes = self.num_can_use_nodes

        # Query the number of available GPUs for applied nodes
        nodesinfo = subprocess.check_output(['squeue', '-p', partion, '--noheader'])
        node_dict = dict()

        for line in nodesinfo.decode().split('\n'):
            if 'gpu:' in line:
                node_info = line.strip().split()
                num_nodes = int(node_info[-3])
                num_gpus = int(node_info[-2].split(':')[-1])
                node_list = node_info[-1]
                if num_nodes == 1:
                    self._add_dict(node_list, num_gpus, node_dict)
                else:
                    avg_gpus = int(num_gpus / num_nodes)
                    result = self._expand_nodelist(node_list)
                    for x in result:
                        self._add_dict(x, avg_gpus, node_dict)

        # Obtain all available idle nodes in the specified partition
        idle_nodes = []
        nodesinfo = subprocess.check_output(['sinfo', '-p', partion, '--noheader'])
        for line in nodesinfo.decode().split('\n'):
            if 'idle' in line:
                node_info = line.strip().split()
                num_nodes = int(node_info[-3])
                node_list = node_info[-1]
                if num_nodes == 1:
                    idle_nodes.append(node_list)
                else:
                    idle_nodes += self._expand_nodelist(node_list)

        # Add idle nodes under resource constraints
        num_allocated_nodes = len(node_dict)
        num_append_nodes = num_can_use_nodes - num_allocated_nodes

        for i, node_ip in enumerate(idle_nodes):
            if i + 1 <= num_append_nodes:
                node_dict[node_ip] = 8

        # Remove nodes with depleted GPUs
        node_dict = {k: v for k, v in node_dict.items() if v != 0}
        return node_dict

    def launch(self, job) -> None:
        """启动 Slurm 作业并管理其执行。

该方法启动指定的 Slurm 作业，并根据同步设置决定是否等待作业完成。如果设置为同步执行，会持续监控作业状态直到完成，然后停止作业。

Args:
    job: 要启动的 SlurmLauncher.Job 对象。

**Returns:**

- 作业的返回值。

Raises:
    AssertionError: 如果传入的 job 不是 SlurmLauncher.Job 类型。
"""
        assert isinstance(job, SlurmLauncher.Job), 'Slurm launcher only support cmd'
        job.start()
        if self.sync:
            while job.status == Status.Running:
                time.sleep(10)
            job.stop()
        return job.return_value
