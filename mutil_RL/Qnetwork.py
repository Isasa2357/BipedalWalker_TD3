import torch
from torch import nn

class mQnet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x) -> torch.Tensor:
        pass

class Qnetwork(mQnet):
    def __init__(self, input_channels: int, output_chanels: int, inner_channels: int=64, inner_layers: int=3):
        super().__init__()
        self.__input_channels = input_channels
        self.__output_channels = output_chanels
        self.__inner_channels = inner_channels
        self.__inner_layers = inner_layers

        self.__layers = nn.ModuleList()

        # 入力層
        self.__layers.append(nn.Linear(self.__input_channels, self.__inner_channels))
        self.__layers.append(nn.ReLU())

        # 隠れ層
        for _ in range(self.__inner_layers):
            self.__layers.append(nn.Linear(self.__inner_channels, self.__inner_channels))
            self.__layers.append(nn.ReLU())
        
        # 出力層
        self.__layers.append(nn.Linear(self.__inner_channels, self.__output_channels))

    def forward(self, x) -> torch.Tensor:
        for layer in self.__layers:
            x = layer(x)
        return x

class DuelingNet(mQnet):
    def __init__(self, input_channels: int, output_channels: int, inner_channels: int=64, inner_layers: int=3):
        super().__init__()
        self._input_channels = input_channels
        self._inner_channels = inner_channels
        self._output_channels = output_channels

        self._inner_layers = inner_layers

        self._layers = nn.ModuleList()

        #### 層定義
        
        # 入力層
        self._layers.append(nn.Linear(self._input_channels, self._inner_channels))
        self._layers.append(nn.ReLU())

        # 隠れ層
        for _ in range(self._inner_layers):
            self._layers.append(nn.Linear(self._inner_channels, self._inner_channels))
            self._layers.append(nn.ReLU())

        # 出力層
        self._base_out = nn.Linear(self._inner_channels, 1)
        self._advantage_out = nn.Linear(self._inner_channels, self._output_channels)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        base = self._base_out(x)
        advantage = self._advantage_out(x)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)  # ←ここ重要
        return base + advantage

        
