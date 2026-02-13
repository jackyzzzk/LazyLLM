import lazyllm
from lazyllm.module import ModuleBase
from lazyllm.common import package


class TencentSearch(ModuleBase):
    """
腾讯搜索接口封装类，用于调用腾讯云的内容搜索服务。

提供对腾讯云搜索API的封装，支持关键词搜索和结果处理。

Args:
    secret_id (str): 腾讯云API密钥ID，用于身份认证
    secret_key (str): 腾讯云API密钥，用于身份认证


Examples:

    from lazyllm.tools.tools import TencentSearch
    secret_id = '<your_secret_id>'
    secret_key = '<your_secret_key>'
    searcher = TencentSearch(secret_id, secret_key)
    """
    def __init__(self, secret_id, secret_key):
        super().__init__()
        from tencentcloud.common.common_client import CommonClient
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        self.cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = 'tms.tencentcloudapi.com'
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.headers = {'X-TC-Action': 'SearchPro'}
        self.common_client = CommonClient(
            'tms', '2020-12-29', self.cred, '', profile=clientProfile)

    def forward(self, query: str):
        """
搜索用户输入的查询。

Args:
    query (str): 用户待查询的内容。

**Returns:**

- package: 包含搜索结果的对象，如果发生错误则返回空package


Examples:

    from lazyllm.tools.tools import TencentSearch
    secret_id = '<your_secret_id>'
    secret_key = '<your_secret_key>'
    searcher = TencentSearch(secret_id, secret_key)
    res = searcher('calculus')
    """
        try:
            res_dict = self.common_client.call_json('SearchPro', {'Query': query, 'Mode': 2}, headers=self.headers)
            res = package(res_dict['Response']['Pages'])
        except Exception as err:
            lazyllm.LOG.error('Request Tencent Search meets error: ', err)
            res = package()
        return res
