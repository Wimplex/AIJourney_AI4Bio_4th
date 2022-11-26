import torch
from torch.autograd import Variable

from nn.nn_blocks.base import PartlyLoadingModel


class BaseEncoder(PartlyLoadingModel):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size

    def get_encoding_size(self, num_channels: int = 1):
        temp = Variable(torch.rand(1, num_channels, self.input_size))
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim