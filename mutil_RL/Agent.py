import os
from copy import copy, deepcopy
import numpy as np

import torch
from torch import nn
from torch import optim

from gymnasium import Env

from ReplayBuffer.Buffer import ReplayBuffer
from ReplayBuffer.ScalarParam import ScalarParam, MultiplyScheduler

from mutil_gpu.mutil_gpu import select_aprop_gpu
from mutil_RL.Qnetwork import mQnet

class DQNAgent:
    def __init__(self, hyperParams: dict, agentParams: dict, 
                 qnet: mQnet, 
                 replayBuf: ReplayBuffer, batch_size: int, 
                 device: str="cpu"):
        # デバイスの設定
        self._device = torch.device(device)

        # ハイパーパラメータ
        self._gamma = hyperParams['gamma']
        self._lr = hyperParams['lr']
        self._epsilon = ScalarParam(hyperParams['epsilon_init'], 
                                    MultiplyScheduler(hyperParams['epsilon_stepsize'], 
                                                      hyperParams['epsilon_destination'], 
                                                      hyperParams['epsilon_init']))
        
        # agentのタイプ
        self._agentType = agentParams['type']

        # 同期タイプ
        self._sync_type = agentParams['sync_type']

        # ネットワークの同期インターバル
        self._qnet_sync_interval = agentParams['sync_interval']
        self._interval_count = 0

        # Qネットワーク
        self._qnet = qnet
        self._target_net = deepcopy(qnet)

        # 最適化，損失関数
        self._optimizer = agentParams['optimizer'](self._qnet.parameters(), lr=self._lr)
        self._loss_func = agentParams['loss_func'](reduction='mean')

        # リプレイバッファ
        self._replayBuf = replayBuf
        self._batch_size = batch_size

        # ログ
        self._loss_history = list()
    
    def get_action(self, state) -> float:
        '''
            actionの選択
        '''
        if np.random.random() < self._epsilon.value():
            return np.random.choice([0, 1])
        else:
            self._qnet.eval()
            with torch.no_grad():
                state = torch.tensor(state, device=self._device).unsqueeze(0)
                qnet_out = self._qnet.forward(state)
                action = qnet_out.argmax().item()
            return action

    def update(self, observation: list) -> None:
        '''
            リプレイバッファへのobservationの追加とネットワークの更新
        '''

        ### リプレイバッファへの追加
        self._replayBuf.add(observation)

        if self._replayBuf.real_size() < self._batch_size:
            return

        ### ネットワークの更新

        # バッチの取り出し
        batch = self._replayBuf.get_batch(self._batch_size)
        status, actions, rewards, next_status, dones = batch
        
        # qnetのQ値の予測値を計算            
        qnet_out = self._qnet.forward(status)
        qnet_pred = qnet_out[range(len(qnet_out)), actions]

        # targetのQ値の予測値を計算
        self._target_net.eval()
        with torch.no_grad():
            target_out = self._target_net.forward(next_status)
            if self._agentType == "DQN":
                next_actions = target_out.argmax(dim=1, keepdim=True).squeeze(1)
            elif self._agentType == "DDQN":
                next_actions = self._qnet.forward(next_status).argmax(dim=1, keepdim=True)
        target_pred = target_out[range(len(target_out)), next_actions]

        target = rewards + (1 - dones) * self._gamma * target_pred
        target = target.detach()

        # lossの計算
        loss = self._loss_func(qnet_pred, target)

        # 学習
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # ネットワークの同期処理
        if self._sync_type == 'step':
            self._interval_count = self.sync_interval_step()

        # 後処理
        self._loss_history.append(loss.item())

    def sync_interval_step(self) -> int:
        '''
            sync_interval(ネットワークの同期インターバル)を進める
            インターバルがqnet_sync_intervalを超えるとネットワークを同期する
        '''
        self._interval_count += 1
        if self._interval_count > self._qnet_sync_interval:
            self._interval_count = 0
            # ネットワークの同期
            self.sync_qnet()
        return self._interval_count
    
    def sync_qnet(self) -> None:
        self._target_net = deepcopy(self._qnet)

    
    @property
    def loss_history(self):
        return self._loss_history
    
    def paramStep(self):
        '''
            Agentのパラメータを進める
        '''
        self._epsilon.step()
    
    def save(self, path, model_name):
        torch.save(self._qnet, os.path.join(path, f'{model_name}.pth'))