# Checkers Reinforcement Learning Project Documentation

## checkers_env.py
### Environment Functions
- `__init__()`: Initialize the checkers board and player
- `initialize_board()`: Create the initial 6x6 board setup
- `reset()`: Reset the board to initial state
- `valid_moves(player)`: Generate valid moves for a given player
- `capture_piece(action)`: Remove captured pieces from the board
- `game_winner(board)`: Determine the winner of the game
- `step(action, player)`: Execute a game move and return next state and reward
- `render()`: Visualize the current board state
- `promote_to_king(board, row, col)`: Promote pieces to kings when they reach opposite end
- `king_valid_moves(board, row, col)`: Generate valid moves for king pieces


## LearningAgent.py (Q-Learning)
### Learning Agent Functions
- `__init__()`: Initialize Q-learning parameters
- `get_state_key(board)`: Convert board state to hashable key
- `evaluation(board)`: Calculate comprehensive reward function
- `choose_action(board)`: Select action using epsilon-greedy strategy
- `q_learning_update(state, action, reward, next_state)`: Update Q-table using Q-learning algorithm
- `learning(num_episodes)`: Train the agent over specified number of episodes

## SARSALearningAgent.py
### SARSA Learning Agent Functions
- Similar structure to Q-Learning agent
- `sarsa_update(state, action, reward, next_state, next_action)`: Update Q-table using SARSA algorithm

## gui.py
### GUI Interaction Functions
- `__init__()`: Initialize game GUI and setup game mode
- `create_board()`: Create the graphical checkers board
- `update_board()`: Refresh board visuals
- `validate_move(start, end)`: Check move validity
- `square_clicked(row, col)`: Handle user interaction with board squares
- `check_game_end()`: Determine game termination

## main.py
### Project Management Functions
- `check_dependencies()`: Verify and install required packages
- `train_model()`: Train the reinforcement learning model
- `run_game()`: Launch the game GUI
- `main()`: Project entry point with mode selection

## `modelpersistance.py`
### Main Functions
- QTable Saving
- QTable Loading
- Listing all saved QTables

## `PerformanceTracker.py`
### Main Functions
- episode Tracking
- performance data saving
- performance data loading

## Key Reinforcement Learning Concepts
- Epsilon-greedy exploration
- Q-table based value estimation
- Reward shaping
- On-policy and off-policy learning