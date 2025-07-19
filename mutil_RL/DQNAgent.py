import torch
from torch import nn
from torch import optim

from gymnasium import Env

from mutil_RL.Qnetwork import Qnetwork
from ReplayBuffer.Buffer import PrioritizedExperienceReplayBuffer
from ReplayBuffer.ScalarParam import *

import copy
import numpy as np
import os
from typing import overload

class DQNAgent:
    def __init__(self, hyper_params: dict, env: Env, 
                 inner_channels: int=32, inner_layers: int=3, optimizer: optim.Optimizer=optim.Adam, loss: nn.Module=nn.MSELoss, 
                 sync_interval_min: int=10, sync_interval_max: int=1000, sync_interval_improve: float=1.005,  
                 buffer_size: int=1000000, state_size: list=None, action_size: list=None, batch_size: int=32, device: str=None):
        # デバイスの設定
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        if device != None:
            self._device = device
            
        # ハイパーパラメータ
        self._gamma = hyper_params['gamma']
        self._lr = hyper_params['lr']
        self._epsilon = hyper_params['epsilon']

        # update interval
        self._sync_interval_max = sync_interval_max
        self._sync_interval_min = sync_interval_min
        self._sync_interval_improve = sync_interval_improve
        self._sync_interval = sync_interval_min
        self._interval_count = 0

        # Qnetへの入力
        self._state_size = env.observation_space.shape[0]
        self._action_size = env.action_space.n

        # Qnet
        self._qnet = Qnetwork(self._state_size, self._action_size, inner_channels, inner_layers=inner_layers).to(device)
        self._target_net = copy.deepcopy(self._qnet).to(device)

        # リプレイバッファ
        self._replay_buf = PrioritizedExperienceReplayBuffer(buffer_size, state_size, action_size, 
                                                             alpha=ScalarParam(0.7, MultiplyScheduler(0.9994, 0.4, 0.7)), 
                                                             beta=ScalarParam(1.0, MultiplyScheduler(1.00091, 1.0, 0.4)), 
                                                             device=device)
        # self._replay_buf = ReplayBuffer(buffer_size)
        
        self._batch_size = batch_size

        # 最適化，損失関数
        self._optimizer = optimizer(self._qnet.parameters(), lr=self._lr)
        self._loss_fn = loss

        # ログ
        self._loss_history = list()
    
    def get_action(self, state) -> int:
        if np.random.rand() < self._epsilon:
            return np.random.choice(self._action_size)
        else:
            self._qnet.eval()
            with torch.no_grad():
                state = torch.tensor(state[np.newaxis, :], device=self._device)
                ret = self._qnet.forward(state)
                return ret.argmax().item()
    
    def update(self, observation: list) -> None:
        '''
        得られたobservationを取得する

        リプレイバッファがバッチサイズ以上であれば，取り出し，パラメータの更新を行う

        observation = [status]
        '''
        self._replay_buf.add(observation)
        self._interval_count += 1

        # モード変更
        self._qnet.train()
        self._target_net.eval()

        # バッチサイズ以下なら終了
        if len(self._replay_buf) < self._batch_size:
            return
        
        ### パラメータの更新

        # バッチの取り出し
        observations, priorities, indics, weights = self._replay_buf.get_batch(self._batch_size)
        status, actions, rewards, next_status, dones = observations
        status.to(self._device)
        actions.to(self._device)
        rewards.to(self._device)
        next_status.to(self._device)
        dones.to(self._device)

        # qnetで推論
        qnet_out = self._qnet.forward(status)
        qnet_pred = qnet_out[np.arange(len(actions)), actions]

        # target_netから教師信号取り出し
        with torch.no_grad(): 
            target_qnet_out = self._target_net.forward(next_status)
        target_pred = target_qnet_out.max(dim=1)[0]

        target = rewards + (1 - dones) * self._gamma * target_pred

        # 損失計算 → パラメータ更新
        loss = self._loss_fn(qnet_pred, target) * weights

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._loss_history.append(loss.item())

        # TD誤差計算
        td_diffs = []
        for reward, tp, qp in zip(rewards, target_pred, qnet_pred):
            td_diffs.append(reward + self._gamma * tp - qp)
        self._replay_buf.update_priorities(indics, td_diffs)

        if self._sync_interval < self._interval_count:
            self.sync_qnet()
    
    def sync_qnet(self):
        self._sync_interval = min(self._sync_interval * self._sync_interval_improve, self._sync_interval_max)
        self._interval_count = 0
        # print(f'sync interval: {self._sync_interval}')
        self._target_net.load_state_dict(self._qnet.state_dict())
    
    def save(self, path, model_name):
        torch.save(self._qnet, os.path.join(path, f'{model_name}.pth'))
    
    def decay_epsilon(self, min_epsilon: float, decay_rate: float):
        self._epsilon = max(min_epsilon, self._epsilon * decay_rate)

    def get_loss_history(self):
        return self._loss_history
    
    def decay_ab(self):
        self._replay_buf.step()
        # print(f'alpha, beta: {self._replay_buf._alpha}, {self._replay_buf._beta}')

class DDQNAgent(DQNAgent):
    def __init__(self, hyper_params: dict, env: Env, 
                 inner_channels: int=32, inner_layers: int=3, optimizer: optim.Optimizer=optim.Adam, loss: nn.Module=nn.MSELoss, 
                 sync_interval_min: int=10, sync_interval_max: int=1000, sync_interval_improve: float=1.005,  
                 buffer_size: int=1000000, state_size: int=None, action_size :int=None, batch_size: int=32, device: str=None):
        super().__init__(hyper_params=hyper_params, env=env, 
                         inner_channels=inner_channels, inner_layers=inner_layers, 
                         optimizer=optimizer, loss=loss, 
                         sync_interval_min=sync_interval_min, sync_interval_max=sync_interval_max, sync_interval_improve=sync_interval_improve, 
                         buffer_size=buffer_size, state_size=state_size, action_size=action_size, batch_size=batch_size, device=device)
    
    def update(self, observation: list) -> None:
        '''
        得られたobservationを取得する

        リプレイバッファがバッチサイズ以上であれば，取り出し，パラメータの更新を行う

        observation = [status]
        '''
        self._replay_buf.add(observation)
        self._interval_count += 1

        # モード変更
        self._qnet.train()
        self._target_net.eval()

        # バッチサイズ以下なら終了
        if len(self._replay_buf) < self._batch_size:
            return
        
        ### パラメータの更新

        # バッチの取り出し
        # self._replay_buf._priorities.show()
        observations, priorities, indics, weights = self._replay_buf.get_batch(self._batch_size)
        status, actions, rewards, next_status, dones = observations
        actions = actions.squeeze(-1)

        # qnetで推論
        qnet_out = self._qnet.forward(status)
        qnet_pred = qnet_out[np.arange(len(actions)), actions]

        # target_netから教師信号取り出し
        with torch.no_grad(): 
            next_action = self._qnet.forward(next_status).argmax(dim=1, keepdim=True)
            target_qnet_out = self._target_net.forward(next_status)
            target_pred = target_qnet_out.gather(1, next_action).squeeze(1)

        target = rewards + (1 - dones) * self._gamma * target_pred

        # 損失計算 → パラメータ更新
        weights = torch.tensor(weights, device=self._device, dtype=qnet_pred.dtype)
        loss = (self._loss_fn(qnet_pred, target) * weights).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # TD誤差計算
        td_diffs = []

        for reward, tp, qp in zip(rewards, target_pred, qnet_pred):
            # print(f'reward: {reward}')
            # print(f'tp: {tp}')
            # print(f'qp: {qp}')
            # print(f'qp shape: {qnet_pred.shape}')
            td_diff = reward + self._gamma * tp - qp
            td_diff = td_diff.detach().abs().cpu()
            td_diffs.append(td_diff)
            

        self._replay_buf.update_priorities(td_diffs, indics)

        self._loss_history.append(loss.item())

        if self._sync_interval < self._interval_count:
            self.sync_qnet()