import time
import random
from queue import Empty
from typing import Callable

from ..core import ComponentBase
import lazyllm
from lazyllm import launchers, flows, LOG
from ...components.utils.file_operate import _image_to_base64, _audio_to_base64, ocr_to_base64

lazyllm.config.add('openai_api', bool, False, 'OPENAI_API', description='Whether to use OpenAI API for vllm deployer.')


class LazyLLMDeployBase(ComponentBase):
    """此类是 ``ComponentBase`` 的一个子类，提供了LazyLLM部署的基础功能。它支持多种媒体类型的编码转换，并提供了结果提取和流式处理的配置选项。

Args:
    launcher (LauncherBase): 用于部署的启动器实例，默认为远程启动器(``launchers.remote()``)。

注意事项: 
    - 继承此类时需要实现具体的部署逻辑
    - 可以通过重写extract_result方法来自定义结果提取逻辑


Examples:
    >>> import lazyllm
    >>> from lazyllm.components.deploy.base import LazyLLMDeployBase
    >>> class MyDeployer(LazyLLMDeployBase):
    ...     def __call__(self, inputs):
    ...         return processed_result
            def extract_result(output, inputs):
    ...         return output.json()['result']
    >>> deployer = MyDeployer()
    >>> result = deployer.extract_result(raw_output, input_data)
    """
    keys_name_handle = None
    message_format = None
    default_headers = {'Content-Type': 'application/json'}
    stream_url_suffix = ''
    stream_parse_parameters = {}

    encoder_map = dict(image=_image_to_base64, audio=_audio_to_base64, ocr_files=ocr_to_base64)

    @staticmethod
    def extract_result(output, inputs):
        """从模型输出中提取最终结果，默认实现直接返回原始输出，子类可重写此方法实现自定义结果提取逻辑。

Args:
    output: 模型原始输出
    inputs: 原始输入数据，可用于结果后处理

**Returns:**

- 处理后的最终结果
"""
        return output

    def __init__(self, *, launcher=launchers.remote()):  # noqa B008
        super().__init__(launcher=launcher)


class DummyDeploy(LazyLLMDeployBase, flows.Pipeline):
    """DummyDeploy(launcher=launchers.remote(sync=False), *, stream=False, **kw)

一个用于测试的模拟部署类，继承自 `LazyLLMDeployBase` 和 `flows.Pipeline`，实现了一个简单的流水线风格部署服务，
支持流式输出（可选）。

该类主要用于内部测试和示例用途。它接收符合 `message_format` 格式的输入，根据是否启用 `stream` 参数，返回
字符串或逐步输出的模拟响应。

Args：
    launcher: 部署器实例，默认值为 `launchers.remote(sync=False)`。
    stream (bool): 是否以流式方式输出结果。
    kw: 其他传递给父类的关键字参数。

Call Arguments:
    keys_name_handle (dict): 输入字段名的映射。 

    message_format (dict): 默认请求模板，包括输入内容与生成参数。 

"""
    keys_name_handle = {'inputs': 'inputs'}
    message_format = {
        'inputs': '',
        'parameters': {
            'do_sample': False,
            'temperature': 0.1,
        }
    }

    def __init__(self, launcher=launchers.remote(sync=False), *, stream=False, **kw):  # noqa B008
        super().__init__(launcher=launcher)

        def func():

            def impl(x):
                LOG.info(f'input is {x["inputs"]}, parameters is {x["parameters"]}')
                return f'reply for {x["inputs"]}, and parameters is {x["parameters"]}'

            def impl_stream(x):
                for s in ['reply', ' for', f' {x["inputs"]}', ', and',
                          ' parameters', ' is', f' {x["parameters"]}']:
                    yield s
                    time.sleep(0.2)
            return impl_stream if stream else impl
        flows.Pipeline.__init__(self, func,
                                lazyllm.deploy.RelayServer(port=random.randint(30000, 40000), launcher=launcher))

    def __call__(self, *args):
        url = flows.Pipeline.__call__(self)
        LOG.info(f'dummy deploy url is : {url}')
        return url

    def __repr__(self):
        return flows.Pipeline.__repr__(self)

def verify_func_factory(error_message: str, running_message: str,  # noqa: C901
                        err_judge: Callable = lambda syb, msg: msg.lstrip().startswith(syb),
                        run_judge: Callable = lambda syb, msg: syb in msg):
    def _hit(symbols, msg, judge):
        return judge(symbols, msg) if isinstance(symbols, str) else any([judge(s, msg) for s in symbols])

    def verify_func(job):
        begin_time = time.time()
        while True:
            try:
                line = job.queue.get(timeout=3)
            except Empty:
                line = ''
                status = job.status
                if status == lazyllm.launchers.status.Failed:
                    LOG.error('[Verify] Service Startup Failed, '
                              'use `export LAZYLLM_EXPECTED_LOG_MODULES=all` for more logs')
                    return False
                LOG.debug(f'[Verify] Timeout when getting log line and current service status: {status}.')
            if _hit(error_message, line, err_judge):
                LOG.error(f'[Verify] Capture error message: {line} \n\n '
                          ', use `export LAZYLLM_EXPECTED_LOG_MODULES=all` for more logs')
                return False
            elif _hit(running_message, line, run_judge):
                LOG.info(f'[Verify] Capture startup message: {line}', name='launcher')
                LOG.success(f'job `{str(job._fixed_cmd).strip()}` executed successfully!')
                break
            if time.time() - begin_time > 600:
                LOG.error('[Verify] Service Startup Timeout, '
                          'use `export LAZYLLM_EXPECTED_LOG_MODULES=all` for more logs')
                return False
        return True
    return verify_func

verify_fastapi_func = verify_func_factory('ERROR:', 'Uvicorn running on')
verify_ray_func = verify_func_factory(['ray.exceptions.RayTaskError', 'Traceback (most recent call last)'],
                                      'Deployed app \'default\' successfully', err_judge=lambda syb, msg: syb in msg)
