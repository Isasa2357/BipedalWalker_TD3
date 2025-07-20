
import numpy as np
from numpy import ndarray


import torch
from torch import nn
from torch.nn import functional as F

from ActorCritic.ACNet import ContinuousDeterministicActorNet, CriticNet
from usefulParam.Param import ScalarParam, makeConstant, makeMultiply
from ReplayBuffer.Buffer import ReplayBuffer, PERBuffer
from mutil_RL.mutil_torch import conv_str2Optimizer, soft_update

class TD3Agent:
    '''
    TD3
    '''

    def __init__(self, 
                 gamma: ScalarParam, lr: ScalarParam, tau: ScalarParam, # ハイパーパラメータ
                 state_size, action_size, # タスクinfo
                 actor_hdn_chnls: int, actor_hdn_lays: int, actor_optimizer: str, # Actor net
                 critic_hdn_chnls: int, critic_hdn_lays: int, cirtic_optimizer: str, # Critic net
                 actor_update_interval: int, critic_update_interval: int, actor_sync_interval: int, critic_sync_interval: int, # sync
                 replayBuf: PERBuffer, batch_size: int, # replayBuf
                 device: torch.device):
        # デバイス
        self._device = device

        # ハイパーパラメータ
        self._gamma = gamma
        self._lr = lr
        self._tau = tau

        # タスクinfo
        self._state_size = state_size
        self._action_size = action_size

        # Actor net
        self._actor = ContinuousDeterministicActorNet(state_size, action_size, actor_hdn_chnls, actor_hdn_lays, 1.0).to(device)
        self._actor_target = ContinuousDeterministicActorNet(state_size, action_size, actor_hdn_chnls, actor_hdn_lays, 1.0).to(device)
        self._actor_optimizer= conv_str2Optimizer(actor_optimizer, self._actor.parameters(), lr=self._lr.value)

        self._actor_update_interval = actor_update_interval
        self._actor_update_interval_count = 0

        self._actor_sync_interval = actor_sync_interval
        self._actor_sync_interval_count = 0

        # Critic net
        self._critic1 = CriticNet(state_size, action_size, critic_hdn_chnls, critic_hdn_lays, 1).to(device)
        self._critic1_target = CriticNet(state_size, action_size, critic_hdn_chnls, critic_hdn_lays, 1).to(device)
        self._critic1_optimizer = conv_str2Optimizer(cirtic_optimizer, self._critic1.parameters(), lr=self._lr.value)

        self._critic2 = CriticNet(state_size, action_size, critic_hdn_chnls, critic_hdn_lays, 1).to(device)
        self._critic2_target = CriticNet(state_size, action_size, critic_hdn_chnls, critic_hdn_lays, 1).to(device)
        self._critic2_optimizer = conv_str2Optimizer(cirtic_optimizer, self._critic2.parameters(), lr=self._lr.value)

        self._critic_lossFunc = nn.MSELoss()

        if type(replayBuf) == PERBuffer:
            self._critic_lossFunc = nn.MSELoss(reduction='none')

        self._critic_update_interval = critic_update_interval
        self._critic_update_interval_count = 0

        self._critic_sync_interval = critic_sync_interval
        self._critic_sync_interval_count = 0

        # Replay Buffer
        self._replayBuf = replayBuf
        self._batch_size = batch_size

        # log
        self._actor_loss_history = list()
        self._critic1_loss_history = list()
        self._critic2_loss_history = list()

        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        
    def get_action(self, state, noise_std=0.1):
        '''
        アクションを選択
        選択は決定的

        Args:
            state[batch_size, state_size]
        Ret:
            action[batch_size, action_size]
        '''
        action = self._actor(state)
        noise = torch.randn_like(action) * noise_std
        return (action + noise).clamp(-1.0, 1.0)
    
    def update(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        '''
        エージェントを更新する
        '''
        # リプレイバッファに経験を追加
        self.add(state, action, reward, next_state, done)

        # 経験をバッチサイズ分取れなければ終了
        if self._replayBuf.real_size < self._batch_size:
            return
        
        # ネットワークを更新
        self._actor_update_interval_count = self.step_count(self._actor_update_interval_count, self._actor_update_interval)
        self._critic_update_interval_count = self.step_count(self._critic_update_interval_count, self._critic_update_interval)
        self._actor_sync_interval_count = self.step_count(self._actor_sync_interval_count, self._actor_sync_interval)
        self._critic_sync_interval_count = self.step_count(self._critic_sync_interval_count, self._critic_sync_interval)
        
        if type(self._replayBuf) == ReplayBuffer:
            status, actions, rewards, next_status, dones = self._replayBuf.get_sample(self._batch_size)
        elif type(self._replayBuf) == PERBuffer:
            status, actions, rewards, next_status, dones, weight = self._replayBuf.get_sample(self._batch_size)

        if self._actor_update_interval_count == 0:
            if type(self._replayBuf) == ReplayBuffer:
                self.update_actor(status)
            elif type(self._replayBuf) == PERBuffer:
                self.update_actor(status, weight)

        if self._critic_update_interval_count == 0:
            if type(self._replayBuf) == ReplayBuffer:
                self.update_critic(status, actions, rewards, next_status, dones)
            elif type(self._replayBuf) == PERBuffer:
                self.update_critic(status, actions, rewards, next_status, dones, weight)

        if self._actor_sync_interval_count == 0:
            self.sync_actor()
        
        if self._critic_sync_interval_count == 0:
            self.sync_critic()
    
    def step_count(self, counter: int, interval: int) -> int:
        return (counter + 1) % interval
    
    def add(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray) -> None:
        self._replayBuf.add(state, action, reward, next_state, done)


    def update_actor(self, status: torch.Tensor, weight: torch.Tensor=torch.empty(0)):
        '''
        actorの更新
        '''
        actions = self._actor.forward(status)
        # print(actions)
        qval1 = self._critic1.forward(status, actions)
        qval2 = self._critic2.forward(status, actions)
        qval = (qval1 + qval2) / 2
        
        # when PER condition, weight qval
        if type(self._replayBuf) == PERBuffer:
            qval *= weight

        actor_loss = -1 * qval.mean()
        
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()
    
    def update_critic(self, status: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_status: torch.Tensor, dones: torch.Tensor, weight: torch.Tensor=torch.empty(0)) -> None:
        '''
        Criticの更新
        '''
        # qvalを計算
        qval1 = self._critic1.forward(status, actions)
        qval2 = self._critic2.forward(status, actions)
        # print("status and actions")
        # print(status)
        # print(actions)
        # print(qval1)

        # next_qvalを計算
        target_noise_std = 0.2
        noise_clip = 0.2
        noise = torch.randn_like(actions) * target_noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        next_actions = (self._actor_target(next_status) + noise).clamp(-1.0, 1.0)
        next_qval1 = self._critic1_target.forward(next_status, next_actions).detach()
        next_qval2 = self._critic2_target.forward(next_status, next_actions).detach()
        next_qval = torch.min(next_qval1, next_qval2)
        target_qval = rewards + (1 - dones) * self._gamma.value * next_qval
        # print(target_qval)
        # print()

        critic1_loss: torch.Tensor = self._critic_lossFunc(qval1, target_qval)
        critic2_loss: torch.Tensor = self._critic_lossFunc(qval2, target_qval)

        if type(self._replayBuf) == PERBuffer:
            critic1_loss *= weight
            critic1_loss = critic1_loss.mean()
            critic2_loss *= weight
            critic2_loss = critic2_loss.mean()

        self._critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self._critic1_optimizer.step()

        self._critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self._critic2_optimizer.step()

        # PERの更新
        if type(self._replayBuf) == PERBuffer:
            critic1_td_diff = torch.abs(target_qval - qval1)
            critic2_td_diff = torch.abs(target_qval - qval2)
            td_diff = torch.min(critic1_td_diff, critic2_td_diff)
            self._replayBuf.update(td_diff)

    def sync_actor(self):
        '''
        Actorとtarget Actorの同期
        '''
        soft_update(self._actor, self._actor_target, self._tau.value)

    def sync_critic(self):
        '''
        Criticとtarget Criticの同期
        '''
        soft_update(self._critic1, self._critic1_target, self._tau.value)
        soft_update(self._critic2, self._critic2_target, self._tau.value)