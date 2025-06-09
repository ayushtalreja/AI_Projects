import unittest
import numpy as np
from checkers_env import checkers_env
from LearningAgent import LearningAgent


class CheckersGameTest(unittest.TestCase):
    def setUp(self):
        self.env = checkers_env()
        self.agent = LearningAgent(env=self.env)

    def test_board_initialization(self):
        """Test initial board setup"""
        self.assertEqual(np.sum(self.env.board > 0), 6)
        self.assertEqual(np.sum(self.env.board < 0), 6)

    def test_valid_moves(self):
        """Test valid moves generation"""
        moves = self.env.valid_moves(1)
        self.assertTrue(len(moves) > 0)

        for move in moves:
            self.assertTrue(len(move) == 4)
            self.assertTrue(all(0 <= x < 6 for x in move))

    def test_game_winner(self):
        """Test game winner determination"""
        # Initial state should be ongoing
        self.assertEqual(self.env.game_winner(self.env.board), 0)

    def test_step_method(self):
        """Test game step method"""
        moves = self.env.valid_moves(1)
        if moves:
            move = moves[0]
            reward = self.env.step(move, 1)
            self.assertIsNotNone(reward)


if __name__ == '__main__':
    unittest.main()