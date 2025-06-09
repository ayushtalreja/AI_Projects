import time
import numpy as np
import random

class SARSALearningAgent:
    def __init__(self, step_size=0.1, epsilon=0.1, discount_factor=0.9, env=None):
        """
        Initialize SARSA Learning Agent for Checkers
        
        :param step_size: Learning rate (alpha)
        :param epsilon: Exploration rate
        :param discount_factor: Gamma value for future rewards
        :param env: Checkers environment
        """
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.env = env
        
        # Initialize Q-table (board state, action) representation
        self.q_table = {} 
    
    def get_state_key(self, board):
        """
        Convert board state to hashable key
        """
        return tuple(map(tuple, board))
    
    def evaluation(self, board):
        """
        Comprehensive reward function
        
        Rewards:
        - Winning: +10
        - Capturing opponent piece: +3
        - Moving forward: +0.5
        - Losing: -10
        - Getting captured: -3
        - Moving backward: -0.5
        - Making a king: +2
        - Losing a king: -2
        """
        reward = 0
        
        # Check game outcome first
        winner = self.env.game_winner(board)
        if winner == self.env.player:
            return 10
        elif winner == -self.env.player:
            return -10
        
        # Count pieces
        player_pieces = np.sum(board > 0)
        opponent_pieces = np.sum(board < 0)
        piece_difference = player_pieces - opponent_pieces
        
        # Positional rewards
        for row in range(6):
            for col in range(6):
                if board[row][col] == self.env.player:
                    # Reward for moving forward
                    if self.env.player == 1 and row > 2:
                        reward += 0.5
                    elif self.env.player == -1 and row < 3:
                        reward += 0.5
        
        # Additional rewards/penalties
        reward += piece_difference * 0.5
        
        return reward
    
    def choose_action(self, board):
        """
        Epsilon-greedy action selection
        """
        valid_moves = self.env.valid_moves(self.env.player)
        
        if not valid_moves:
            return None
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Exploitation
        state_key = self.get_state_key(board)
        
        # Initialize Q-values for unseen state
        if state_key not in self.q_table:
            self.q_table[state_key] = {tuple(move): 0 for move in valid_moves}
        
        # Choose action with highest Q-value
        q_values = self.q_table[state_key]
        return max(valid_moves, key=lambda move: q_values.get(tuple(move), 0))
    
    def sarsa_update(self, state, action, reward, next_state, next_action):
        """
        SARSA learning update with robust next action handling
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {tuple(action): 0}
        
        # Handle case where next_action is None
        if next_action is None:
            next_q = 0
        else:
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {tuple(next_action): 0}
            
            next_q = self.q_table[next_state_key].get(tuple(next_action), 0)
        
        # Get current state-action Q-values
        current_q = self.q_table[state_key].get(tuple(action), 0)
        
        # SARSA update rule
        updated_q = current_q + self.step_size * (
            reward + self.discount_factor * next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state_key][tuple(action)] = updated_q
    
    def learning(self, num_episodes=1000):
        """
        Training method for the agent
        """
        print("Starting training...")
        total_reward = 0  # Initialize total reward
        for episode in range(num_episodes):
            # Reset environment
            self.env.reset()
            current_board = self.env.board.copy()
            
            # Choose initial action
            current_action = self.choose_action(current_board)
            
            while current_action is not None:
                # Simulate action
                self.env.step(current_action, self.env.player)
                next_board = self.env.board.copy()
                
                # Evaluate reward
                reward = self.evaluation(next_board)
                total_reward += reward  # Accumulate rewards
                
                # Choose next action
                next_action = self.choose_action(next_board)
                
                # Perform SARSA update
                self.sarsa_update(current_board, current_action, 
                                  reward, next_board, next_action)
                
                # Update for next iteration
                current_board = next_board
                current_action = next_action
                
                # Check game end
                if self.env.game_winner(current_board) != 0:
                    break
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
        return total_reward  # Return total reward