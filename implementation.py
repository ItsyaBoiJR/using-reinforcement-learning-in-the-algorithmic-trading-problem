import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state):
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        policy_logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return policy_logits, value, hidden_state

# Environment simulation (dummy environment for demonstration)
class DummyTradingEnv:
    def __init__(self, time_steps=100):
        self.time_steps = time_steps
        self.current_step = 0
        self.state_dim = 5  # Example: [price, volume, indicator1, indicator2, indicator3]
        self.action_space = 3  # [Buy, Hold, Sell]
        self.state = np.random.randn(self.state_dim)

    def reset(self):
        self.current_step = 0
        self.state = np.random.randn(self.state_dim)
        return self.state

    def step(self, action):
        reward = np.random.randn()  # Random reward for demonstration
        self.current_step += 1
        done = self.current_step >= self.time_steps
        self.state = np.random.randn(self.state_dim)  # Random next state
        return self.state, reward, done

# A3C Training Loop
def train_a3c(env, model, optimizer, gamma=0.99, max_episodes=100):
    for episode in range(max_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        hidden_state = (torch.zeros(1, 1, model.lstm.hidden_size),
                        torch.zeros(1, 1, model.lstm.hidden_size))
        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            policy_logits, value, hidden_state = model(state, hidden_state)
            probs = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        advantages = returns - values.squeeze()

        # Compute loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{max_episodes}, Loss: {loss.item():.4f}, Total Reward: {sum(rewards):.2f}")

if __name__ == '__main__':
    # Hyperparameters
    input_dim = 5
    hidden_dim = 128
    action_dim = 3
    learning_rate = 0.001
    max_episodes = 10

    # Initialize environment, model, and optimizer
    env = DummyTradingEnv()
    model = ActorCritic(input_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_a3c(env, model, optimizer, max_episodes=max_episodes)