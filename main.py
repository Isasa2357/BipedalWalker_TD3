from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import gymnasium as gym

import torch

from ReplayBuffer.Buffer import ReplayBuffer, PERBuffer
from ActorCritic.TD3 import TD3Agent
from usefulParam.Param import makeConstant, makeMultiply, makeLinear

def main():
    env = gym.make("BipedalWalker-v3")
    # env = gym.make("Pendulum-v1")
    
    state_size = 24
    action_size = 4

    device = torch.device('cuda')

    replayBuf = ReplayBuffer(1000000, state_size, action_size, device=device)
    # replayBuf = PERBuffer(1000000, makeConstant(0.5), makeLinear(1.0, 0.4, 8000), state_size, action_size, device=device)
    agent = TD3Agent(makeConstant(0.99), makeConstant(0.0003), makeConstant(0.02), 
                     state_size, action_size, 
                     256, 3, "Adam", 
                     256, 3, "Adam", 
                     8, 4, 15, 5, 
                     replayBuf, 64, device)
    
    for _ in tqdm(range(5000)):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = np.random.random(action_size) * 2.0 - 1.0
            # print(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = np.array(terminated or truncated)
            reward = np.array(reward)

            agent.add(state, action, reward, next_state, done)

            state = next_state

    reward_history = list()
    episodes = 10000
    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        
        done = False
        sum_reward = 0.0

        while not done:
            # 行動の選択
            action = agent.get_action(torch.tensor(state, device=device))
            action = action.detach().cpu().numpy()
            # print(action, type(action))

            # 環境を進める
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = np.array(terminated or truncated)
            reward = np.array(reward)

            # エージェントの更新
            agent.update(state, action, reward, next_state, done)

            # 後処理
            state = next_state
            sum_reward += reward
        reward_history.append(sum_reward)
        tqdm.write(f'episode: {episode}, reward: {sum_reward}')
    
    plt.plot(reward_history)
    plt.show()

if __name__ == '__main__':
    main()