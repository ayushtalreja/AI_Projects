# Checkers Reinforcement Learning Project

## Overview
This project implements a Checkers game using Reinforcement Learning techniques, specifically the Qlearning and SARSA algorithms. The game is played on a 6x6 board with advanced machine learning capabilities.

## Features
- 6x6 Checkers board
- QLearning Reinforcement Learning Algorithm
- SARSA Reinforcement Learning Algorithm
- Flexible game modes
  - Human vs Computer
  - Computer vs Computer
- Sophisticated reward function
- Move validation and game state tracking

## Game Modes
1. Human vs Computer
2. Computer vs Computer

## Learning Agent Details
- Uses SARSA algorithm
- Epsilon-greedy exploration
- Comprehensive reward function
- Adaptable learning parameters

## Project Structure
- `checkers_env.py`: Game environment
- `LearningAgent.py`: QLearning RL agent implementation
- `SARSALearningAgent.py`: Sarsa RL agent implementation
- `gui.py`: Graphical user interface
- `modelpersistance.py`: Logic for model saving and retrieving
- `PerformanceTracker.py`: Logic for performance tracking such as episodes rewards
- `CheckersGameTest.py`: Test for checkers environment
- `main.py`: Game execution script

## Customization
Modify learning parameters in `LearningAgent`:
- `step_size`: Learning rate
- `epsilon`: Exploration rate
- `discount_factor`: Future reward consideration

## Contributors
- Ayush Kumar
- Moaaz Eid

## License
MIT License