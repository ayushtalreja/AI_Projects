import tkinter as tk
from tkinter import messagebox

import numpy as np

from checkers_env import checkers_env
from LearningAgent import LearningAgent
from modelpersistance import ModelPersistence


class HumanVsTrainedAgentGUI:
    """
    A Tkinter-based GUI for letting a human (Player 1) play against a
    trained Q-learning agent (Player -1).  We load the Q-table from disk,
    so the AI can use its learned policy.
    """

    def __init__(self, master, q_table_path='q_table/checkers_q_table.pkl'):
        self.master = master
        self.master.title("Checkers: Human vs Trained Agent")

        # Environment
        self.env = checkers_env()

        # Current player: human is +1, AI is -1
        self.current_player = 1

        # Load a Q-trained agent for the AI
        self.ai_agent = LearningAgent(env=self.env, epsilon=0.0)  # Epsilon=0 -> no random moves
        loaded = ModelPersistence.load_q_table(self.ai_agent, q_table_path)
        if loaded:
            print(f"Loaded Q-table from: {q_table_path}")
        else:
            print("WARNING: Could not load Q-table. The AI will start untrained!")

        # GUI state
        self.selected_piece = None
        self.game_over = False

        # Create the board
        self.squares = []
        self.create_board()

        # Status label (bottom)
        self.status_label = tk.Label(master, text="Your turn (Player 1)", font=('Arial', 12))
        self.status_label.grid(row=6, columnspan=6)

    def create_board(self):
        """
        Create a 6x6 checkerboard of Tkinter Buttons and display initial positions.
        """
        for row in range(6):
            row_squares = []
            for col in range(6):
                square = tk.Button(
                    self.master, width=5, height=2,
                    command=lambda r=row, c=col: self.square_clicked(r, c)
                )
                square.grid(row=row, column=col)
                row_squares.append(square)
            self.squares.append(row_squares)

        self.update_board()  # Show initial positions

    def update_board(self):
        """
        Update the text and color of each square based on the environment's board state.
        """
        for row in range(6):
            for col in range(6):
                piece = self.env.board[row][col]
                bg_color = 'light gray' if (row + col) % 2 == 0 else 'dark gray'
                text_color = 'black'
                text = ''

                if piece == 1:
                    text = '🔵'  # P1 piece
                    text_color = 'blue'
                elif piece == -1:
                    text = '🔴'  # P2 piece (the AI)
                    text_color = 'red'
                elif piece == 2:
                    text = '👑'  # King for P1
                    text_color = 'gold'
                elif piece == -2:
                    text = '🤴🏽'  # King for AI
                    text_color = 'purple'

                self.squares[row][col].configure(
                    text=text,
                    fg=text_color,
                    bg=bg_color
                )

    def square_clicked(self, row, col):
        """
        Human clicks on a square.  Handle piece selection and moves for Player 1.
        If the move is valid and completed, let the AI move next (if game not over).
        """
        if self.game_over or self.current_player != 1:
            return  # Ignore clicks if the game is over or it's not the human's turn

        if self.selected_piece is None:
            # First click: pick which piece to move
            piece = self.env.board[row][col]
            # Only allow picking your own piece
            if piece > 0:
                self.selected_piece = (row, col)
                self.highlight_moves(row, col)
        else:
            # Second click: try to move the selected piece to (row, col)
            start_row, start_col = self.selected_piece
            if self.validate_move(start_row, start_col, row, col):
                # Perform the move
                action = [start_row, start_col, row, col]
                self.env.step(action, self.current_player)
                self.selected_piece = None
                self.reset_highlights()
                self.update_board()
                self.check_game_end()

                # If game not over, switch to AI's turn
                if not self.game_over:
                    self.current_player = -1
                    self.status_label.config(text="AI's turn (Player -1)...")
                    self.master.after(500, self.ai_move)  # Delay briefly, then AI moves
            else:
                # Invalid move, just reset selection
                self.selected_piece = None
                self.reset_highlights()

    def ai_move(self):
        """
        Let the trained Q-learning agent choose and perform an action for Player -1.
        Then switch back to the human if the game is not over.
        """
        if self.game_over or self.current_player != -1:
            return

        # AI picks an action from its Q-table
        action = self.ai_agent.choose_action(self.env.board)

        # If no valid moves, that means AI is stuck; game over
        if action is None:
            self.game_over = True
            self.status_label.config(text="No moves left for the AI. You win!")
            messagebox.showinfo("Game Over", "No moves left for the AI. You win!")
            return

        self.env.step(action, self.current_player)
        self.update_board()
        self.check_game_end()

        if not self.game_over:
            # Switch back to human
            self.current_player = 1
            self.status_label.config(text="Your turn (Player 1)")

    def highlight_moves(self, row, col):
        """
        Highlight valid move destinations for the piece at (row,col).
        """
        valid_moves = self.env.valid_moves(self.current_player)
        for move in valid_moves:
            sr, sc, er, ec = move
            if sr == row and sc == col:
                self.squares[er][ec].configure(bg='lightgreen')

        # Also highlight the piece itself in yellow
        self.squares[row][col].configure(bg='yellow')

    def reset_highlights(self):
        """
        Reset the board colors to their original checkered pattern.
        """
        for r in range(6):
            for c in range(6):
                bg_color = 'light gray' if (r + c) % 2 == 0 else 'dark gray'
                self.squares[r][c].configure(bg=bg_color)

    def validate_move(self, sr, sc, er, ec):
        """
        Check if (sr,sc) -> (er,ec) is a valid move for the current player.
        """
        valid_moves = self.env.valid_moves(self.current_player)
        return [sr, sc, er, ec] in valid_moves

    def check_game_end(self):
        """
        Check if the game has ended by either a winner or a draw.
        """
        winner = self.env.game_winner(self.env.board)
        if winner != 0 or self.env.is_draw():
            self.game_over = True
            if winner == 1:
                msg = "Game Over - You win!"
            elif winner == -1:
                msg = "Game Over - The AI wins!"
            else:
                msg = "Game Over - It's a draw!"
            self.status_label.config(text=msg)
            messagebox.showinfo("Game Over", msg)
            
