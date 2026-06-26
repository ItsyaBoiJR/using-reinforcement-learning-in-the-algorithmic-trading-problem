# Reinforcement Learning for Algorithmic Trading

This repository contains the implementation of the research paper [Using Reinforcement Learning in the Algorithmic Trading Problem](https://arxiv.org/pdf/2002.11523v1) by Evgeny Ponomarev, Ivan Oseledets, and Andrzej Cichocki. The paper explores the application of reinforcement learning (RL) techniques to algorithmic trading, treating trading as a Markov Decision Process (MDP) and leveraging the Asynchronous Advantage Actor-Critic (A3C) method for optimizing trading strategies.

---

## Core Concept

Algorithmic trading involves designing automated systems to make trading decisions in financial markets. This paper proposes using reinforcement learning to model trading as a sequential decision-making problem. By framing trading as a game with states, actions, and rewards:

- **States** represent the current market conditions and portfolio status.
- **Actions** correspond to trading decisions, such as buying, selling, or holding.
- **Rewards** quantify the profitability of the actions, taking into account trading commissions.

The authors employ the **A3C algorithm**, a popular reinforcement learning method, to train trading agents. A3C uses both actor and critic networks to simultaneously learn a policy (actor) and value function (critic). The experiments investigate the use of different neural network architectures, including recurrent layers, to capture the temporal dependencies in financial data.

---

## Key Results

The RL-based trading system was evaluated on real anonymized financial data, specifically using the RTS Index futures (MOEX:RTSI). The best-performing model achieved:

- **Profitability**: 66% per annum (after accounting for trading commissions).
- **Effectiveness**: Demonstrated the potential of reinforcement learning in real-world financial trading scenarios.

---

## Repository Overview

This repository contains the Python implementation of the A3C-based trading system described in the paper. The code is written using **PyTorch** and is structured to facilitate experimentation with different neural network architectures and trading strategies.

### Features

1. **A3C Algorithm Implementation**:
   - Asynchronous training with multiple agents interacting with the environment in parallel.
   - Actor and critic networks for policy optimization and value estimation.
   
2. **Environment for Algorithmic Trading**:
   - Simulates trading on financial instruments using real-world data.
   - Defined states, actions, and rewards for reinforcement learning.

3. **Neural Network Architectures**:
   - Support for feedforward and recurrent layers to model temporal dependencies in financial data.
   - Configurable network designs for experimentation.

4. **Performance Metrics**:
   - Evaluation of trading strategies in terms of profitability and robustness.
   - Includes commission costs in the reward calculation.

---

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.0+
- Required Python packages (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Installation

Clone the repository:
```bash
git clone https://github.com/evgps/a3c_trading.git
cd a3c_trading
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. **Prepare Your Data**:
   - Replace the placeholder dataset with your own financial time series data.
   - Ensure the data is formatted correctly for the trading environment.

2. **Train the Model**:
   Run the training script to start learning an optimal trading strategy:
   ```bash
   python train.py
   ```
   Customize hyperparameters and neural network architecture in the configuration file.

3. **Evaluate the Model**:
   Test the trained model on validation or unseen data:
   ```bash
   python evaluate.py
   ```

4. **Experimentation**:
   Modify the environment, network architectures, or RL parameters to explore different configurations.

---

## File Structure

- `train.py`: Script for training the A3C model.
- `evaluate.py`: Script for evaluating the trained model.
- `environment.py`: Implementation of the trading environment.
- `models.py`: Definitions of neural network architectures (feedforward, recurrent, etc.).
- `utils.py`: Utility functions for data processing and logging.
- `requirements.txt`: List of required Python packages.

---

## Results and Discussion

The provided code allows users to replicate the experiments described in the paper and test the A3C algorithm on their own financial data. The inclusion of recurrent layers can help capture patterns in time-series data, potentially improving trading performance.

---

## References

- [Using Reinforcement Learning in the Algorithmic Trading Problem](https://arxiv.org/pdf/2002.11523v1)
- [Original GitHub Repository](http://github.com/evgps/a3c_trading)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or feedback, please contact the authors or open an issue on the [GitHub repository](https://github.com/evgps/a3c_trading).