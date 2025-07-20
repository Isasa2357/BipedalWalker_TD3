import os
from copy import copy, deepcopy
import numpy as np
from numpy import ndarray

import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.nn import functional as F

from typing import List, Tuple, Type, Callable

from ReplayBuffer.Buffer import ReplayBuffer

from usefulParam.Param import *

from mutil_RL.mutil_torch import factory_LinearReLU_ModuleList, factory_LinearReLU_Sequential, conv_str2Optimizer, hard_update, soft_update

######################################## Actor ########################################

class ContinuousDeterministicActorNet(nn.Module):
    '''
    連続値タスク用 決定論的ポリシー ActorNet
    '''

    def __init__(self, state_size: int, action_size: int, hdn_chnls: int, hdn_lays: int, action_scale: float):
        super().__init__()
        self._action_scale = action_scale

        self._network = factory_LinearReLU_Sequential(state_size, hdn_chnls, hdn_lays, action_size)
    
    def forward(self, status: torch.Tensor) -> torch.Tensor:
        '''
        推論

        args:
            status[batch_size, state_size]
        Ret:
            [batch, action_size]
        '''
        ret = self._network.forward(status)
        return self._action_scale * torch.tanh(ret)

class ContinuousStochasticActorNet(nn.Module):
    '''
    連続値タスク用 確率的ポリシー ActorNet
    '''

    def __init__(self, state_size: int, action_size: int, hdn_chnls: int, hdn_lays: int, action_scale: float):
        super().__init__()

        self._action_scale= action_scale
        self._log_std_min = -20
        self._log_std_max = 2

        self._network = factory_LinearReLU_Sequential(state_size, hdn_chnls, hdn_lays, hdn_chnls)
        self._mu_out_lay = nn.Linear(hdn_chnls, action_size)
        self._log_std_lay = nn.Linear(hdn_chnls, action_size)

    def forward(self, status: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        推論

        Args:
            status[batch_size, state_size]
        Ret:
            action: サンプリングされたアクション
            log_prob: サンプリングされたアクションの対数確率
            mu: アクションの分布の平均
            log_std: サンプリングに使用したアクションの対数標準偏差
        '''

        # ネットワークを使った推論
        x = self._network(status)
        mu = self._mu_out_lay(x)
        log_std = self._log_std_lay(x)

        # アクションのサンプリング
        std = torch.exp(log_std)
        action_normal = torch.distributions.Normal(mu, std)
        z = action_normal.rsample()
        action = torch.tanh(z) * self._action_scale

        # アクションの選択確率を計算(計算応報が理解できない)
        log_prob = action_normal.log_prob(z).sum(dim=-1)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - z - F.softplus(-2*z))).sum(dim=-1)


        return action, log_prob, mu, log_std



######################################## Critic ##################################################

class CriticNet(nn.Module):
    '''
        Q値推定 Critic
    '''

    def __init__(self, 
                 state_size: int, action_size: int, 
                 hdn_chnls: int, hdn_layers: int, out_chnls: int):
        super().__init__()

        # 変数定義
        self._action_size = action_size
        self._state_size = state_size
        self._in_chnls = self._action_size + self._state_size
        self._hdn_chnls = hdn_chnls
        self._hdn_layers = hdn_layers
        self._out_chnls = out_chnls

        # ネットワーク定義
        self._network = factory_LinearReLU_Sequential(self._in_chnls, self._hdn_chnls, self._hdn_layers, self._out_chnls)
    
    def forward(self, status: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # 入力調整
        if len(status.shape) == 1:
            status = status.unsqueeze(1)

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        # comm
        x = torch.cat([status, actions], dim=1)
        x = self._network.forward(x)
        
        return x