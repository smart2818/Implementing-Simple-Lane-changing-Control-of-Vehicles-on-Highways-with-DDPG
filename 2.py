import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.tensor as ts
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import gymnasium as gym
from gymnasium import spaces
import math as ma


SPEED_UNIT = "kmph"
# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x))

        return x

# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    class DDPG:
        def __init__(self, state_dim, action_dim, max_action):
            self.actor = Actor(state_dim, action_dim, max_action)
            self.actor_target = Actor(state_dim, action_dim, max_action)
            self.critic = Critic(state_dim, action_dim)
            self.critic_target = Critic(state_dim, action_dim)

            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

            self.replay_buffer = deque(maxlen=1000000)
            self.batch_size = 128
            self.gamma = 0.99
            self.tau = 0.001

        def select_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state).detach().numpy()[0]

            return action

        def store_transition(self, state, action, reward, next_state, done):
            self.replay_buffer.append((state, action, reward, next_state, done))

        def update(self):
            if len(self.replay_buffer) < self.batch_size:
                return

            batch = np.random.choice(self.replay_buffer, self.batch_size)
            state, action, reward, next_state, done = zip(*batch)

            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor(reward).unsqueeze(1)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done).unsqueeze(1)

            # 计算目标Q值
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (1 - done) * self.gamma * target_Q

            # 更新Critic网络
            current_Q = self.critic(state, action)
            critic_loss = nn.MSELoss()(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 更新Actor网络
            actor_loss = -self.critic(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新目标网络
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

                class HighwayEnv(gym.Env):
                    def __init__(self):
                        super(HighwayEnv, self).__init__()

                        # 遵循国际单位制
                        self.dt = 0.1
                        self.speed = 16.67
                        self.current_speed = 16.67
                        self.accelerate = 25
                        self.decelerate = 10

                        self.current_x_pos = 0
                        self.current_y_pos = 0
                        self.current_z_pos = 0.75

                        self.direction = 1
                        self.left = -1
                        self.right = 1
                        self.current_steering_angle = math.radians(90)
                        self.next_left_position = [self.next_x_position + self.current_speed * self.dt * self.left * math.sin(self.current_steering_angle),self.next_y_position + self.current_speed * self.dt * self.left * math.cos(self.current_steering_angle),]
                        self.next_right_position = [
                            self.next_x_position + self.current_speed * self.dt * self.right * math.sin(
                                self.current_steering_angle),
                            self.next_y_position + self.current_speed * self.dt * self.right * math.cos(
                                self.current_steering_angle), ]

                        self.current_position = [self.current_x_pos, self.current_y_pos]
                        self.max_acceleration = 5.0
                        self.state_max_acceleration = 5.0
                        self.Maximum_Braking_Distance = 16
                        self.Minimum_safe_distance_between_vehicles = 30
                        self.Current_steering_angle = math.radians(0)
                        self.Maximum_steering_angle = math.radians(35)
                        self.Maximum_action_steering_angle = math.radians(35)

                        space = gym.spaces.Box(low=[self.current_x_pos - self.width / 2, self.current_y_pos - self.length / 2, self.current_z_pos - self.height / 2], high=[self.current_x_pos - self.width / 2, self.current_y_pos - self.length / 2, self.current_z_pos - self.height / 2], dtype=float)
                        self.length = 4
                        self.width = 2
                        self.height = 1.5
                        center_x = (space.high[0] + space.low[0]) / 2
                        center_y = (space.high[1] + space.low[1]) / 2
                        center_z = (space.high[2] + space.low[2]) / 2

                        # 定义状态维度
                        self.current_state_dim = [self.current_position, self.speed, self.state_max_acceleration, self.Maximum_Braking_Distance, self.Minimum_safe_distance_between_vehicles]
                        # 定义动作维度
                        self.action_dim = [self.current_speed,self.accelerate,self.decelerate,self.Maximum_action_steering_angle]
                        # 定义最大动作值
                        self.max_action = [self.max_acceleration,self.Maximum_steering_angle]

                        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
                        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action,
                                                       shape=(self.action_dim,))

                    def step(self, action):
                        # 执行动作，返回下一个状态、奖励、是否结束以及其他信息
                        self.next_state = self.next_right_position if self.direction == 1 else self.next_left_position
                        self.reward = 1 if
                        done = ...
                        info = ...

                        return next_state, reward, done, info

                    def reset(self):
                        # 重置环境，返回初始状态
                        initial_state = ...

                        return initial_state

                    def render(self):
                        # 这里实现环境的可视化，具体根据你的需求绘制高速公路、车辆等场景
                        plt.figure()
                        plt.plot(...)  # 示例，绘制相关元素
                        plt.show()

                    def step(self, action):
                        # 执行动作，返回下一个状态、奖励、是否结束以及其他信息
                        next_state = ...
                        reward = ...
                        done = ...
                        info = ...

                        return next_state, reward, done, info

                    def reset(self):
                        # 重置环境，返回初始状态
                        initial_state = ...

                        return initial_state

                    def render(self):
                        # 这里实现环境的可视化，具体根据你的需求绘制高速公路、车辆等场景
                        plt.figure()
                        plt.plot(...)  # 示例，绘制相关元素
                        plt.show()

                        def on_play_clicked(event):
                            global running, paused
                            paused = False
                            running = True

                        def on_pause_clicked(event):
                            global paused
                            paused = True

                        def on_fast_forward_clicked(event):
                            global fast_forward
                            fast_forward = True

                        def on_fast_backward_clicked(event):
                            global fast_backward
                            fast_backward = True

                        # 创建可视化窗口
                        fig, ax = plt.subplots()
                        ax.set_title("Highway Vehicle Lane Changing Control")

                        # 创建按钮
                        play_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
                        pause_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
                        fast_forward_ax = plt.axes([0.58, 0.05, 0.1, 0.075])
                        fast_backward_ax = plt.axes([0.46, 0.05, 0.1, 0.075])

                        play_button = Button(play_ax, 'Play')
                        pause_button = Button(pause_ax, 'Pause')
                        fast_forward_button = Button(fast_forward_ax, 'Fast Forward')
                        fast_backward_button = Button(fast_backward_ax, 'Fast Backward')

                        play_button.on_clicked(on_play_clicked)
                        pause_button.on_clicked(on_pause_clicked)
                        fast_forward_button.on_clicked(on_fast_forward_clicked)
                        fast_backward_button.on_clicked(on_fast_backward_clicked)

                        running = False
                        paused = True
                        fast_forward = False
                        fast_backward = False



if __name__ == "__main__":
    env = HighwayEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action

    agent = DDPG(state_dim, action_dim, max_action)

    num_episodes = 1000
    max_steps_per_episode = 100

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            if running:
                if fast_forward:
                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.store_transition(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                    time.sleep(0.01)  # 可根据需要调整快进速度
                elif fast_backward:
                    # 这里可以实现快退逻辑，比如从回放缓冲区中获取之前的状态等
                    pass
                elif not paused:
                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.store_transition(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward

                    if done:
                        break

            env.render()

        agent.update()

        print(f"Episode {episode}: Reward = {episode_reward}")