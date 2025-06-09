import pickle
import numpy as np
from checkers_env import checkers_env
from LearningAgent import LearningAgent

class SelfPlay:
    def __init__(self, q_table_path='q_table/checkers_q_table.pkl'):
        """
        Initialize self play environment with two agents using the same Q-table
        
        Args:
            q_table_path (str): Path to the saved Q-table file
        """
        # Initialize environment
        self.env = checkers_env()
        
        # Load Q-table
        with open(q_table_path, 'rb') as f:
            q_table = pickle.load(f)
        
        # Initialize both agents with the same Q-table
        self.agent1 = LearningAgent(step_size=0.1, epsilon=0.05, env=self.env)
        self.agent2 = LearningAgent(step_size=0.1, epsilon=0.05, env=self.env)
        
        self.agent1.q_table = q_table
        self.agent2.q_table = q_table
        
        # Initialize statistics
        self.stats = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0
        }
    
    def play_single_game(self):
        """
        Play a single game between the two agents
        
        Returns:
            int: 1 for player 1 win, -1 for player 2 win, 0 for draw
        """
        self.env.reset()
        current_player = 1
        
        while True:
            # Get current state
            current_state = self.env.board.copy()
            
            # Get action from current player's agent
            if current_player == 1:
                action = self.agent1.choose_action(current_state)
            else:
                action = self.agent2.choose_action(current_state)
            
            # If no valid moves, current player loses
            if action is None:
                return -current_player
            
            # Make move
            self.env.step(action, current_player)
            
            # Check for winner
            winner = self.env.game_winner(self.env.board)
            if winner != 0:
                return winner
            
            # Switch players
            current_player *= -1
    
    def run_self_play(self, num_games=1000):
        """
        Run multiple self-play games and save results
        
        Args:
            num_games (int): Number of games to play
        """
        for game in range(num_games):
            result = self.play_single_game()
            
            # Update statistics
            if result == 1:
                self.stats['player1_wins'] += 1
            elif result == -1:
                self.stats['player2_wins'] += 1
            else:
                self.stats['draws'] += 1
            
            # Print progress
            if (game + 1) % 100 == 0:
                print(f"Completed {game + 1} games")
        
        self.save_results()
    
    def save_results(self):
        """
        Save the self-play results to a file
        """
        total_games = sum(self.stats.values())
        results = {
            'total_games': total_games,
            'player1_wins': self.stats['player1_wins'],
            'player2_wins': self.stats['player2_wins'],
            'draws': self.stats['draws'],
            'player1_win_rate': self.stats['player1_wins'] / total_games,
            'player2_win_rate': self.stats['player2_wins'] / total_games,
            'draw_rate': self.stats['draws'] / total_games
        }
        
        np.savez('performance/self_play_results.npz', **results)
        
        # Also save as human-readable text
        with open('self_play_results.txt', 'w') as f:
            f.write("Self Play Results\n")
            f.write("================\n\n")
            f.write(f"Total Games: {total_games}\n")
            f.write(f"Player 1 Wins: {self.stats['player1_wins']} ({results['player1_win_rate']:.2%})\n")
            f.write(f"Player 2 Wins: {self.stats['player2_wins']} ({results['player2_win_rate']:.2%})\n")
            f.write(f"Draws: {self.stats['draws']} ({results['draw_rate']:.2%})\n")
            