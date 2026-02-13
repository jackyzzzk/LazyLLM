import os
import json
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func
from .utils import get_log_path, make_log_dir, parse_options_keys

lazyllm.config.add('default_embedding_engine', str, '', 'DEFAULT_EMBEDDING_ENGINE',
                   description='The default embedding engine to use.')

class Infinity(LazyLLMDeployBase):
    """此类是 ``LazyLLMDeployBase`` 的子类，基于 [Infinity](https://github.com/michaelfeil/infinity) 框架提供的高性能文本嵌入、重排序和CLIP等能力。

Args:
    launcher (lazyllm.launcher): Infinity 的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    launcher (Launcher, optional): 启动器配置，默认为remote(ngpus=1)。
    model_type (str, optional): 模型类型，默认为'embed'。
    log_path (str, optional): 日志文件路径，默认为None。
    **kw: 额外的配置参数，包括host、port、batch-size等。



Examples:
    >>> import lazyllm
    >>> from lazyllm import deploy
    >>> deploy.Infinity()
    <lazyllm.llm.deploy type=Infinity>
    """
    keys_name_handle = {
        'inputs': 'input',
    }
    message_format = {
        'input': 'who are you ?',
    }
    default_headers = {'Content-Type': 'application/json'}
    target_name = 'embeddings'

    def __init__(self, launcher=launchers.remote(ngpus=1), model_type='embed', log_path=None, **kw):  # noqa B008
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'host': '0.0.0.0',
            'port': None,
            'batch-size': 256,
        })
        self._model_type = model_type
        kw.pop('stream', '')
        # Infinity (embedding model) doesn't support tensor parallel, ignore 'tp' parameter
        kw.pop('tp', None)
        self.options_keys = kw.pop('options_keys', [])
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] else True
        self.temp_folder = make_log_dir(log_path, 'lmdeploy') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                            f'base_model({base_model}) will be used')
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)
            cmd = f'infinity_emb v2 --model-id {finetuned_model} --no-bettertransformer '
            if isinstance(self._launcher, launchers.EmptyLauncher) and self._launcher.ngpus:
                available_gpus = self._launcher._get_idle_gpus()
                required_count = self._launcher.ngpus
                if required_count <= len(available_gpus):
                    try:
                        use_cuda_visible = lazyllm.config['cuda_visible']
                    except (KeyError, AttributeError):
                        use_cuda_visible = False
                    if use_cuda_visible:
                        # Use logical GPU IDs (0, 1, 2...) when CUDA_VISIBLE_DEVICES is set
                        gpu_ids = ','.join(map(str, range(required_count)))
                    else:
                        # Use physical GPU IDs when CUDA_VISIBLE_DEVICES is not set
                        gpu_ids = ','.join(map(str, available_gpus[:required_count]))
                    cmd += f'--device-id={gpu_ids} '
                else:
                    raise RuntimeError(
                        f'Insufficient GPUs available (required: {required_count}, '
                        f'available: {len(available_gpus)})'
                    )
            cmd += self.kw.parse_kwargs()
            cmd += ' ' + parse_options_keys(self.options_keys)
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        """获取Infinity服务的URL地址。根据部署模式和作业状态，返回对应的API访问URL地址。

Args:
    job (Optional[Any]): 作业对象，如果为None则使用当前实例的job属性。
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return f'http://<ip>:<port>/{self.target_name}'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/{self.target_name}'

    @staticmethod
    def extract_result(x, inputs):
        """从Infinity API响应中提取结果数据。
解析Infinity服务的JSON响应，根据返回的对象类型提取嵌入向量或重排序结果。

Args:
    x (str): API返回的JSON字符串响应。
    inputs (Dict): 原始输入数据，用于确定返回结果的格式。
"""
        try:
            res_object = json.loads(x)
        except Exception as e:
            LOG.warning(f'JSONDecodeError on load {x}')
            raise e
        assert 'object' in res_object
        object_type = res_object['object']
        if object_type == 'list':  # for infinity >= 0.0.64
            object_type = res_object['data'][0]['object']
        if object_type == 'embedding':
            res_list = [item['embedding'] for item in res_object['data']]
            if len(res_list) == 1 and type(inputs['input']) is str:
                res_list = res_list[0]
            return json.dumps(res_list)
        elif object_type == 'rerank':
            return [(x['index'], x['relevance_score']) for x in res_object['results']]

class InfinityRerank(Infinity):
    keys_name_handle = {'inputs': 'query'}
    message_format = {'query': 'who are you ?', 'documents': ['string'], 'return_documents': False,
                      'raw_scores': False, 'top_n': 1, 'model': 'default/not-specified'}
    default_headers = {'Content-Type': 'application/json'}
    target_name = 'rerank'
