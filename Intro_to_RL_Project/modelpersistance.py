import pickle
import os
import json

class ModelPersistence:
    @staticmethod
    def save_q_table(agent, filename='q_table/checkers_q_table.pkl'):
        """
        Save Q-table to a pickle file
        """
        with open(filename, 'wb') as f:
            pickle.dump(agent.q_table, f)
        
        # Save metadata
        metadata = {
            'step_size': agent.step_size,
            'epsilon': agent.epsilon,
            'discount_factor': agent.discount_factor
        }
        
        with open(f'{filename}_metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    @staticmethod
    def load_q_table(agent, filename='q_table/checkers_q_table.pkl'):
        """
        Load Q-table from a pickle file
        """
        try:
            with open(filename, 'rb') as f:
                agent.q_table = pickle.load(f)
            
            # Load metadata
            with open(f'{filename}_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            agent.step_size = metadata['step_size']
            agent.epsilon = metadata['epsilon']
            agent.discount_factor = metadata['discount_factor']
            
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filename}")
            return False
    
    @staticmethod
    def list_saved_models(directory='.'):
        """
        List available saved models
        """
        q_tables = [f for f in os.listdir(directory) if f.endswith('q_table.pkl')]
        return q_tables