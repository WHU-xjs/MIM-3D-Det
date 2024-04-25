from mmcv.runner import BaseModule

class BaseEncoder(BaseModule):
    def __init__(self, stream_name="base"):
        super(BaseEncoder, self).__init__()
        self.stream_name = stream_name

    def loss(self, input_data, gt_labels, metas=None):
        pass