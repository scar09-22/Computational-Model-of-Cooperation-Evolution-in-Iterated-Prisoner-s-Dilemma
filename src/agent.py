import random

class EvolvableAgent: 
    def __init__(self, genome=None):
        if genome is None:
            self.genome = [random.choice(['C', 'D']) for _ in range(4)]
        else:
            self.genome = genome
        
        self.fitness = 0
        self.history = []
    
    def reset(self):
        self.fitness = 0
        self.history = []
    
    def make_move(self, opponent_history):
        if not self.history:
            return self.genome[0]
        
        last_my_move = self.history[-1]
        last_opponent_move = opponent_history[-1]
        
        if last_my_move == 'C' and last_opponent_move == 'C':
            return self.genome[0]
        if last_my_move == 'C' and last_opponent_move == 'D':
            return self.genome[1]
        if last_my_move == 'D' and last_opponent_move == 'C':
            return self.genome[2]
        if last_my_move == 'D' and last_opponent_move == 'D':
            return self.genome[3]
    
    def get_name(self):
        return ''.join(self.genome)
