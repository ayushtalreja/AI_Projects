import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime


class PerformanceTracker:
    def __init__(self, agent):
        self.agent = agent
        self.episode_rewards = []
        self.episode_wins = []
        self.cumulative_learning_curve = []
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Training time tracking
        self.training_start_time = None
        self.training_end_time = None
        self.episode_times = []
        self.total_training_time = 0

        # Performance tracking
        self.win_history = []
        self.loss_history = []
        self.draw_history = []
        self.win_rate_history = []

    def start_training(self):
        """
        Mark the start of training
        """
        self.training_start_time = time.time()

    def end_training(self):
        """
        Mark the end of training and calculate total time
        """
        self.training_end_time = time.time()
        self.total_training_time = self.training_end_time - self.training_start_time

    def track_episode(self, episode, total_reward, is_win):
        """
        Track performance metrics for each episode
        """
        print(f"Tracking Episode {episode}: Reward={total_reward}, Win={is_win}")

        self.episode_rewards.append(total_reward or 0)  # Ensure non-None reward
        self.episode_wins.append(1 if is_win else 0)

        # Compute cumulative average
        if len(self.episode_rewards) > 0:
            cumulative_avg = np.mean(self.episode_rewards)
            self.cumulative_learning_curve.append(cumulative_avg)
        else:
            self.cumulative_learning_curve.append(0)

        # Update win rate history
        if len(self.episode_wins) > 0:
            win_rate = sum(self.episode_wins) / len(self.episode_wins)
            self.win_rate_history.append(win_rate)
        else:
            self.win_rate_history.append(0)

    def plot_learning_progress(self):
        """
        Visualize learning progress with enhanced metrics
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Rewards Plot
        ax1.plot(self.episode_rewards, label='Episode Rewards')
        ax1.set_title('Episode Rewards Over Training')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()

        # Cumulative Average Plot
        ax2.plot(self.cumulative_learning_curve, label='Cumulative Average Reward', color='green')
        ax2.set_title('Cumulative Average Reward')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.legend()

        # Win Rate Plot
        ax3.plot(self.win_rate_history, label='Win Rate', color='blue')
        ax3.set_title('Win Rate Over Training')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate')
        ax3.legend()

        # Game Outcomes
        outcomes = ['Wins', 'Losses', 'Draws']
        values = [self.wins, self.losses, self.draws]
        bars = ax4.bar(outcomes, values, color=['green', 'red', 'blue'])
        ax4.set_title('Game Outcomes')
        ax4.set_ylabel('Count')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        plt.tight_layout()

        # Add training time information
        if self.total_training_time > 0:
            time_text = f'Total Training Time: {self.format_time(self.total_training_time)}'
            fig.suptitle(time_text, y=1.02)

        plt.show()

    def format_time(self, seconds):
        """
        Format time in seconds to a readable string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def save_performance_data(self, filename='performance/performance_data.npz'):
        """
        Save enhanced performance metrics
        """
        np.savez(filename,
                 rewards=self.episode_rewards,
                 wins=self.episode_wins,
                 cumulative_curve=self.cumulative_learning_curve,
                 episode_times=self.episode_times,
                 win_rate_history=self.win_rate_history,
                 total_training_time=self.total_training_time,
                 game_outcomes={
                     'wins': self.wins,
                     'losses': self.losses,
                     'draws': self.draws
                 })

    def load_performance_data(self, filename='performance/performance_data.npz'):
        """
        Load enhanced performance data
        """
        data = np.load(filename, allow_pickle=True)
        self.episode_rewards = data['rewards'].tolist()
        self.episode_wins = data['wins'].tolist()
        self.cumulative_learning_curve = data['cumulative_curve'].tolist()
        self.episode_times = data['episode_times'].tolist()

        self.win_rate_history = data['win_rate_history'].tolist()
        self.total_training_time = float(data['total_training_time'])

        # Load game outcomes
        outcomes = data['game_outcomes'].item()
        self.wins = outcomes['wins']
        self.losses = outcomes['losses']
        self.draws = outcomes['draws']

    def update_results(self, result):
        """
        Update the outcome counts and histories
        """
        if result == 1:  # Win
            self.wins += 1
        elif result == -1:  # Loss
            self.losses += 1
        elif result == 0.5:  # Draw
            self.draws += 1

    def plot_results(self):
        """
        Plot enhanced game outcomes visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart of outcomes
        outcomes = ['Wins', 'Losses', 'Draws']
        values = [self.wins, self.losses, self.draws]
        bars = ax1.bar(outcomes, values, color=['green', 'red', 'blue'])
        ax1.set_title("Game Outcomes")
        ax1.set_ylabel("Count")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        # Win rate over time
        total_games = len(self.win_history) + len(self.loss_history) + len(self.draw_history)
        if total_games > 0:
            games = range(1, total_games + 1)
            win_rate = [w for _ , w in enumerate(self.win_history, 1)]
            ax2.plot(games, win_rate, label='Win Rate', color='green')
            ax2.set_title('Win Rate Over Time')
            ax2.set_xlabel('Games Played')
            ax2.set_ylabel('Win Rate')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()