import json
import os
import uuid
import requests
from typing import Tuple, List, Dict, Union
from urllib.parse import urljoin
import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    OnlineChatModuleBase,
    LazyLLMOnlineEmbedModuleBase, LazyLLMOnlineRerankModuleBase,
    LazyLLMOnlineSTTModuleBase, LazyLLMOnlineText2ImageModuleBase
)
from ..fileHandler import FileHandlerBase
from lazyllm.thirdparty import zhipuai
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.components.formatter import encode_query_with_filepaths


class GLMChat(OnlineChatModuleBase, FileHandlerBase):
    """GLMChat 类，继承自 OnlineChatModuleBase 和 FileHandlerBase，封装了对智谱 GLM 系列模型的在线调用功能。  
支持对话生成、文件处理以及模型微调等能力。默认使用 GLM-4 模型，也可指定其他训练型模型（如 chatglm3-6b、chatglm_12b 等）。

Args:
    base_url (Optional[str]): 智谱 GLM 服务的 API 接口地址，默认为 "https://open.bigmodel.cn/api/paas/v4/"。
    model (Optional[str]): 使用的 GLM 模型名称，默认为 "glm-4"，也可选择 TRAINABLE_MODEL_LIST 中的其他模型。
    api_key (Optional[str]): 访问 GLM 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (Optional[bool]): 是否开启流式输出，默认为 True。
    return_trace (Optional[bool]): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的可选参数。
"""
    TRAINABLE_MODEL_LIST = ['chatglm3-6b', 'chatglm_12b', 'chatglm_32b', 'chatglm_66b', 'chatglm_130b']
    VLM_MODEL_PREFIX = ['glm-4.5v', 'glm-4.1v', 'glm-4v']
    MODEL_NAME = 'glm-4'

    def __init__(self, base_url: str = 'https://open.bigmodel.cn/api/paas/v4/', model: str = None,
                 api_key: str = None, stream: str = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['glm_api_key'],
                         model_name=model or lazyllm.config['glm_model_name'] or GLMChat.MODEL_NAME,
                         base_url=base_url, stream=stream, return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        self.default_train_data = {
            'model': None,
            'training_file': None,
            'validation_file': None,
            'extra_hyperparameters': {
                'fine_tuning_method': None,  # lora\full, default: lora,
                'fine_tuning_parameters': {
                    'max_sequence_length': None  # [1, 8192](int), default: 8192
                }
            },
            'hyperparameters': {
                'learning_rate_multiplier': 0.01,  # (0,5] , default: 1.0
                'batch_size': None,  # [1, 32], default: 8
                'n_epochs': 1,  # [1, 10], default: 3
            },
            'suffix': None,
            'request_id': None
        }
        self.fine_tuning_job_id = None

    def _get_system_prompt(self):
        return ('You are ChatGLM, an AI assistant developed based on a language model trained by Zhipu AI. '
                'Your task is to provide appropriate responses and support for user\'s questions and requests.')

    def _get_models_list(self):
        return ['glm-4', 'glm-4v', 'glm-3-turbo', 'chatglm-turbo', 'cogview-3', 'embedding-2', 'text-embedding']

    def _convert_file_format(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as fr:
            dataset = [json.loads(line) for line in fr]

        json_strs = []
        for ex in dataset:
            lineEx = {'messages': []}
            messages = ex.get('messages', [])
            for message in messages:
                role = message.get('role', '')
                content = message.get('content', '')
                if role in ['system', 'user', 'assistant']:
                    lineEx['messages'].append({'role': role, 'content': content})
            json_strs.append(json.dumps(lineEx, ensure_ascii=False))

        return '\n'.join(json_strs)

    def _upload_train_file(self, train_file):
        url = urljoin(self._base_url, 'files')
        self.get_finetune_data(train_file)

        file_object = {
            'purpose': (None, 'fine-tune', None),
            'file': (os.path.basename(train_file), self._dataHandler, 'application/json')
        }

        with requests.post(url, headers=self._get_empty_header(), files=file_object) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            # delete temporary training file
            self._dataHandler.close()
            return r.json()['id']

    def _update_kw(self, data, normal_config):
        cur_data = self.default_train_data.copy()
        cur_data.update(data)

        cur_data['extra_hyperparameters']['fine_tuning_method'] = normal_config['finetuning_type'].strip().lower()
        cur_data['extra_hyperparameters']['fine_tuning_parameters']['max_sequence_length'] = normal_config['cutoff_len']
        cur_data['hyperparameters']['learning_rate_multiplier'] = normal_config['learning_rate']
        cur_data['hyperparameters']['batch_size'] = normal_config['batch_size']
        cur_data['hyperparameters']['n_epochs'] = normal_config['num_epochs']
        cur_data['suffix'] = str(uuid.uuid4())[:7]
        return cur_data

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = urljoin(self._base_url, 'fine_tuning/jobs')
        data = {'model': train_model, 'training_file': train_file_id}
        if len(kw) > 0:
            if 'finetuning_type' in kw:
                data = self._update_kw(data, kw)
            else:
                data.update(kw)

        with requests.post(url, headers=self._header, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()['id']
            self.fine_tuning_job_id = fine_tuning_job_id
            status = self._status_mapping(r.json()['status'])
            return (fine_tuning_job_id, status)

    def _cancel_finetuning_job(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            return 'Invalid'
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = os.path.join(self._base_url, f'fine_tuning/jobs/{job_id}/cancel')
        with requests.post(fine_tune_url, headers=self._header) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        status = r.json()['status']
        if status == 'cancelled':
            return 'Cancelled'
        else:
            return f'JOB {job_id} status: {status}'

    def _query_finetuned_jobs(self):
        fine_tune_url = os.path.join(self._base_url, 'fine_tuning/jobs/')
        with requests.get(fine_tune_url, headers=self._get_empty_header()) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()

    def _get_finetuned_model_names(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        model_data = self._query_finetuned_jobs()
        res = list()
        for model in model_data['data']:
            res.append([model['id'], model['fine_tuned_model'], self._status_mapping(model['status'])])
        return res

    def _status_mapping(self, status):
        if status == 'succeeded':
            return 'Done'
        elif status == 'failed':
            return 'Failed'
        elif status == 'cancelled':
            return 'Cancelled'
        elif status == 'running':
            return 'Running'
        else:  # create, validating_files, queued
            return 'Pending'

    def _query_job_status(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError('No job ID specified. Please ensure that a valid "fine_tuning_job_id" is '
                               'provided as an argument or started a training job.')
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        _, status = self._query_finetuning_job(job_id)
        return self._status_mapping(status)

    def _get_log(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError('No job ID specified. Please ensure that a valid "fine_tuning_job_id" is '
                               'provided as an argument or started a training job.')
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = os.path.join(self._base_url, f'fine_tuning/jobs/{job_id}/events')
        with requests.get(fine_tune_url, headers=self._get_empty_header()) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return job_id, r.json()

    def _get_curr_job_model_id(self):
        if not self.fine_tuning_job_id:
            return None, None
        model_id, _ = self._query_finetuning_job(self.fine_tuning_job_id)
        return self.fine_tuning_job_id, model_id

    def _query_finetuning_job_info(self, fine_tuning_job_id):
        fine_tune_url = os.path.join(self._base_url, f'fine_tuning/jobs/{fine_tuning_job_id}')
        with requests.get(fine_tune_url, headers=self._get_empty_header()) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        status = info['status']
        fine_tuned_model = info['fine_tuned_model'] if 'fine_tuned_model' in info else None
        return (fine_tuned_model, status)

    def _query_finetuning_cost(self, fine_tuning_job_id):
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        if 'trained_tokens' in info and info['trained_tokens']:
            return info['trained_tokens']
        else:
            return None

    def _create_deployment(self) -> Tuple[str]:
        return (self._model_name, 'RUNNING')

    def _query_deployment(self, deployment_id) -> str:
        return 'RUNNING'


class GLMEmbed(LazyLLMOnlineEmbedModuleBase):
    """GLM嵌入模型接口类，用于调用智谱AI的文本嵌入服务。

Args:
    embed_url (str): 嵌入服务API地址，默认为"https://open.bigmodel.cn/api/paas/v4/embeddings"
    embed_model_name (str): 嵌入模型名称，默认为"embedding-2"
    api_key (str): API密钥
"""
    def __init__(self,
                 embed_url: str = 'https://open.bigmodel.cn/api/paas/v4/embeddings',
                 embed_model_name: str = 'embedding-2',
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        super().__init__(embed_url, api_key or lazyllm.config['glm_api_key'], embed_model_name,
                         batch_size=batch_size, **kw)


class GLMRerank(LazyLLMOnlineRerankModuleBase):
    """智谱AI的重排序模块，继承自OnlineEmbeddingModuleBase，用于对文档进行相关性重排序。

Args:
    embed_url (str): 重排序API的基础URL，默认为"https://open.bigmodel.cn/api/paas/v4/rerank"。
    embed_model_name (str): 使用的模型名称，默认为"rerank"。
    api_key (str): 智谱AI的API密钥，如果未提供则从lazyllm.config['glm_api_key']读取。

属性：
    type: 返回模型类型，固定为"ONLINE_RERANK"。

主要功能：
    - 对输入的查询和文档列表进行相关性重排序
    - 支持自定义排序参数
    - 返回每个文档的相关性得分
"""
    def __init__(self,
                 embed_url: str = 'https://open.bigmodel.cn/api/paas/v4/rerank',
                 embed_model_name: str = 'rerank',
                 api_key: str = None, **kw):
        super().__init__(embed_url, api_key or lazyllm.config['glm_api_key'], embed_model_name, **kw)

    @property
    def type(self):
        return 'RERANK'

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            'query': query,
            'documents': documents,
            'top_n': top_n,
            'return_documents': False,
            'return_raw_scores': True
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Tuple]:
        return [(result['index'], result['relevance_score']) for result in response['results']]


class GLMMultiModal():
    """智谱AI的多模态基础模块，继承自OnlineMultiModalBase，用于处理多模态任务。

Args:
    model_name (str): 模型名称。
    api_key (str): API密钥，如果未提供则从lazyllm.config['glm_api_key']读取。
    base_url (str): API的基础URL，默认为'https://open.bigmodel.cn/api/paas/v4'。
    return_trace (bool): 是否返回调用追踪信息，默认为False。
    **kwargs: 其他传递给基类的参数。

功能特点：

    1. 支持多模态输入处理
    2. 使用ZhipuAI客户端进行API调用
    3. 提供统一的多模态接口
    4. 可自定义基础URL和API密钥

注意：
    该类作为GLM多模态功能的基础类，通常作为其他具体多模态实现（如语音转文本、文本生成图像等）的父类。
"""
    def __init__(self, api_key: str = None, base_url: str = 'https://open.bigmodel.cn/api/paas/v4'):
        api_key = api_key or lazyllm.config['glm_api_key']
        self._client = zhipuai.ZhipuAI(api_key=api_key, base_url=base_url)


class GLMSTT(LazyLLMOnlineSTTModuleBase, GLMMultiModal):
    """GLM语音识别模块，继承自GLMMultiModal。

提供基于智谱AI的语音转文本(STT)功能，支持音频文件的语音识别。

Args:
    model_name (str, optional): 模型名称，默认为配置中的模型名或"glm-asr"
    api_key (str, optional): API密钥，默认为配置中的密钥
    return_trace (bool, optional): 是否返回追踪信息，默认为False
    **kwargs: 其他模型参数
"""
    MODEL_NAME = 'glm-asr'

    def __init__(self, model_name: str = None, api_key: str = None,
                 base_url: str = 'https://open.bigmodel.cn/api/paas/v4',
                 return_trace: bool = False, **kwargs):
        super().__init__(model_name=model_name or GLMSTT.MODEL_NAME, api_key=api_key, return_trace=return_trace,
                         base_url=base_url, **kwargs)
        GLMMultiModal.__init__(self, api_key=api_key, base_url=base_url)

    def _forward(self, files: List[str] = [], url: str = None, model: str = None, **kwargs):  # noqa B006
        assert len(files) == 1, 'GLMSTT only supports one file'
        assert os.path.exists(files[0]), f'File {files[0]} not found'
        client = self._client
        if url and url != getattr(self, '_base_url', None):
            client = zhipuai.ZhipuAI(api_key=self._api_key, base_url=url)
        transcriptResponse = client.audio.transcriptions.create(
            model=model,
            file=open(files[0], 'rb'),
        )
        return transcriptResponse.text


class GLMText2Image(LazyLLMOnlineText2ImageModuleBase, GLMMultiModal):
    """GLM文本生成图像模块，继承自 GLMMultiModal，封装了调用 GLM CogView-4 模型生成图像的功能。  
支持根据文本提示（prompt）生成指定数量和分辨率的图像，并可通过 API Key 调用远程服务。

Args:
    model_name (Optional[str]): 使用的 GLM 模型名称，默认使用 "cogview-4-250304" 或配置中的 'glm_text_to_image_model_name'。
    api_key (Optional[str]): API Key，用于访问 GLM 图像生成服务。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 GLMMultiModal 的参数。
"""
    MODEL_NAME = 'cogview-4-250304'

    def __init__(self, model_name: str = None, api_key: str = None, return_trace: bool = False,
                 base_url: str = 'https://open.bigmodel.cn/api/paas/v4', **kwargs):
        super().__init__(model_name=model_name or GLMText2Image.MODEL_NAME, api_key=api_key,
                         return_trace=return_trace, base_url=base_url, **kwargs)
        GLMMultiModal.__init__(self, api_key=api_key, base_url=base_url)
        if self._type == LLMType.IMAGE_EDITING:
            raise ValueError('GLM series models do not support image editing now.')

    def _forward(self, input: str = None, n: int = 1, size: str = '1024x1024',
                 url: str = None, model: str = None, **kwargs):
        runtime_url = url or self._base_url
        runtime_model = model or self._model_name
        client = self._client
        if runtime_url and runtime_url != getattr(self, '_base_url', None):
            client = zhipuai.ZhipuAI(api_key=self._api_key, base_url=runtime_url)
        call_params = {
            'model': runtime_model,
            'prompt': input,
            'n': n,
            'size': size,
            **kwargs
        }
        response = client.images.generations(**call_params)
        return encode_query_with_filepaths(None, bytes_to_file([requests.get(result.url).content
                                                                for result in response.data]))
