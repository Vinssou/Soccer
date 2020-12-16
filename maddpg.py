import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_agent import Agent 

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.95            # discount factor

EXPLORATION_DECAY = 0.9999999
EXPLORATION_MIN = 0.01
LEARN_FREQUENCY = 20
LEARN_COUNT = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MaddpgAgent: 

    def __init__(self, agent_count, state_size, action_size, random_seed):

        self.action_size = action_size
        self.state_size = state_size
        self.agent_count = agent_count
        self.agents = [Agent(agent_count, state_size, action_size, random_seed) for _ in range(agent_count) ]

        random.seed(random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.exploration = 1.0

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.setp_count = 0


    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        self.setp_count += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.setp_count % LEARN_FREQUENCY == 0:
            for _ in range(LEARN_COUNT):
                for idx in range(len(self.agents)):
                    experiences = self.memory.sample()
                    self.learn(idx, experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for idx, state in enumerate(states):
            state = torch.from_numpy(state).float().to(device)
            state = state.unsqueeze(0)
            self.agents[idx].actor_local.eval()
            with torch.no_grad():
                action = self.agents[idx].actor_local(state).cpu().data.numpy()
            self.agents[idx].actor_local.train()
            if add_noise:
                noise = self.exploration * self.noise.sample()
                action += noise
                self.exploration *= EXPLORATION_DECAY
                self.exploration = max(EXPLORATION_MIN, self.exploration)
            actions.append(action)

        actions = np.array(actions)
        actions = actions.squeeze()
        return actions

    def target_act(self, states):
        """Returns actions for given state as per current policy."""
        actions = torch.tensor([])
        assert len(states) == self.agent_count
        for idx, state in enumerate(states):
            state = torch.reshape(state, (BATCH_SIZE, self.state_size))
            action = self.agents[idx].actor_target(state)
            max_index = torch.argmax(action, 1)
            action = torch.zeros((BATCH_SIZE, self.action_size))
            for i in range(BATCH_SIZE):
                action[i, max_index[i]] = 1.0
            actions = torch.cat((actions, action))
        actions = torch.reshape(actions, (self.agent_count, BATCH_SIZE, self.action_size))
        actions = actions.permute(1, 0, 2)
        actions = torch.reshape(actions, (BATCH_SIZE, self.agent_count * self.action_size))
        return actions


    def local_act(self, states):
        actions = torch.tensor([])
        assert len(states) == self.agent_count
        for idx, state in enumerate(states):
            state = torch.reshape(state, (BATCH_SIZE, self.state_size))
            action = self.agents[idx].actor_local(state)
            max_index = torch.argmax(action, 1)
            action = torch.zeros((BATCH_SIZE, self.action_size))
            for i in range(BATCH_SIZE):
                action[i, max_index[i]] = 1.0
            actions = torch.cat((actions, action))
        actions = torch.reshape(actions, (self.agent_count, BATCH_SIZE, self.action_size))
        actions = actions.permute(1, 0, 2)
        actions = torch.reshape(actions, (BATCH_SIZE, self.agent_count * self.action_size))
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, agent_index, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        all_next_states = np.reshape(next_states, (BATCH_SIZE, self.agent_count * self.state_size))
        all_states = np.reshape(states, (BATCH_SIZE, self.agent_count * self.state_size))
        all_actions = np.reshape(actions, (BATCH_SIZE, self.agent_count * self.action_size))

        next_states = np.swapaxes(next_states, 0, 1)
        states = np.swapaxes(states, 0, 1)
        rewards = np.swapaxes(rewards, 0, 1)
        dones = np.swapaxes(dones, 0, 1)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_act(next_states)
        with torch.no_grad():
            Q_targets_next = self.agents[agent_index].critic_target(all_next_states, actions_next)
        Q_targets_next.squeeze_()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[agent_index] + (gamma * Q_targets_next * (1 - dones[agent_index]))

        # Compute critic loss
        Q_expected = self.agents[agent_index].critic_local(all_states, all_actions)
        Q_expected.squeeze_()

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.agents[agent_index].critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.agents[agent_index].critic_local.parameters(), 1)
        self.agents[agent_index].critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.local_act(states)
        actor_loss = -self.agents[agent_index].critic_local(all_states, actions_pred).mean()
        # Minimize the loss
        self.agents[agent_index].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agents[agent_index].actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.agents[agent_index].soft_update()

    def save(self, name):
        for idx, agent in enumerate(self.agents):
            agent.save(name + str(idx))

    def load(self, name):
        for idx, agent in enumerate(self.agents):
            agent.load(name + str(idx))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.3, sigma=0.4):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


