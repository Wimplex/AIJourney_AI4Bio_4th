import torch

from nn.nn_blocks.base import PartlyLoadingModel


class BaseHead(PartlyLoadingModel):
    def __init__(self, input_size: int, output_size: int, curricular: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._curricular = curricular

    @property
    def curricular(self):
        return self._curricular

    def forward(self, x: torch.Tensor, emb: bool = False) -> torch.Tensor:
        raise NotImplementedError()