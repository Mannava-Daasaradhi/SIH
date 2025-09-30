import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time
from torch.utils.tensorboard import SummaryWriter

# Import the environment we created
from train_env import TrainTrafficEnv

# --- Part 1: Replay Memory ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Part 2: The Neural Network (The "Brain") ---
class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Part 3: The DQN Agent ---
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.n_trains = len(env.train_configs)
        self.n_actions = 2 ** self.n_trains
        self.n_observations = len(env.observation_space.sample())
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.policy_net = QNetwork(self.n_observations, self.n_actions)
        self.target_net = QNetwork(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                action_index = self.policy_net(state).max(1)[1].view(1, 1)
        else:
            action_index = torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)
        
        action_array = [int(x) for x in list(np.binary_repr(action_index.item(), width=self.n_trains))]
        return action_index, np.array(action_array)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None # Return None if not optimizing
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item() # Return the loss value

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)


# --- Part 4: The Training Loop with TensorBoard Logging ---
if __name__ == "__main__":
    # Initialize TensorBoard
    writer = SummaryWriter("runs/dqn_railway_experiment_1")

    env = TrainTrafficEnv()
    agent = DQNAgent(env)
    
    num_episodes = 500
    start_time = time.time()
    episode_rewards = []

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        total_loss = 0
        optim_steps = 0

        for t in range(200): # Max steps per episode
            action_index, action_array = agent.select_action(state)
            observation, reward, terminated, _, info = env.step(action_array)
            total_reward += reward
            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            agent.memory.push(state, action_index, next_state, reward_tensor)
            state = next_state
            
            loss = agent.optimize_model()
            if loss is not None:
                total_loss += loss
                optim_steps += 1

            agent.update_target_network()

            if terminated:
                break
        
        episode_rewards.append(total_reward)

        # Log metrics to TensorBoard at the end of each episode
        avg_loss = total_loss / optim_steps if optim_steps > 0 else 0
        writer.add_scalar('Reward/Episode', total_reward, i_episode)
        writer.add_scalar('Loss/Episode', avg_loss, i_episode)
        writer.add_scalar('Metrics/Throughput', info['metrics']['throughput'], i_episode)
        writer.add_scalar('Metrics/Avg_Journey_Time_Steps', info['metrics']['avg_journey_time'], i_episode)
        writer.add_scalar('Metrics/Total_Wait_Time_Steps', info['metrics']['total_wait_time'], i_episode)

        if (i_episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"Episode {i_episode+1}/{num_episodes} | Avg Reward (last 10): {avg_reward:.2f}")

    end_time = time.time()
    print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")
    
    writer.close()
    torch.save(agent.policy_net.state_dict(), "dqn_policy_net.pth")
    print("Model saved to dqn_policy_net.pth")