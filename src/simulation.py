from config import PAYOFF

class Simulation:
    def __init__(self, rounds_per_match):
        self.rounds_per_match = rounds_per_match
    
    def _play_match(self, agent1, agent2):
        agent1.reset()
        agent2.reset()
        
        for _ in range(self.rounds_per_match):
            move1 = agent1.make_move(agent2.history)
            move2 = agent2.make_move(agent1.history)
            
            score1, score2 = PAYOFF[(move1, move2)]
            agent1.fitness += score1
            agent2.fitness += score2
            
            agent1.history.append(move1)
            agent2.history.append(move2)
    
    def run_tournament(self, agents):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                self._play_match(agents[i], agents[j])
