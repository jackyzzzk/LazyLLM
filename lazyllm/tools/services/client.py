from lazyllm.launcher import Status

class ClientBase(object):
    """客户端基类，用于管理服务连接和状态转换。

Args:
    url (str): 服务端点的URL地址。

属性：
    url: 服务端点的URL地址。
"""
    def __init__(self, url):
        self.url = url

    def uniform_status(self, status):
        """统一化任务状态字符串。

Args:
    status (str): 原始状态字符串。

**Returns:**

- str: 标准化的状态字符串，可能的值包括：
    - 'Invalid': 无效状态
    - 'Ready': 就绪状态
    - 'Done': 完成状态
    - 'Cancelled': 已取消状态
    - 'Failed': 失败状态
    - 'Running': 运行中状态
    - 'Pending': 等待中状态（包括TBSubmitted、InQueue、Pending）
"""
        if status == 'Invalid':
            res = 'Invalid'
        elif status == 'Ready':
            res = 'Ready'
        elif Status[status] == Status.Done:
            res = 'Done'
        elif Status[status] == Status.Cancelled:
            res = 'Cancelled'
        elif Status[status] == Status.Failed:
            res = 'Failed'
        elif Status[status] == Status.Running:
            res = 'Running'
        else:  # TBSubmitted, InQueue, Pending
            res = 'Pending'
        return res
