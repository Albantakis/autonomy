import numpy as np
import copy

class Block:
    '''
        THE FOLLOWING FUNCTIONS ARE MOSTLY FOR VISUALIZING OR RUNNING THE
        COMPLEXIPHI WORLD IN PYTHON (FOR CHECKING CONSISTENCY) AND ARE NOT
        WELL COMMENTED. FUNCTIONS USEFUL FOR ANALYSIS ARE COMMENTED.
    '''
    def __init__(self, size, direction, block_type, ini_x, ini_y=0):
        self.size = size
        self.direction = direction
        self.type = block_type
        self.x = ini_x
        self.y = ini_y

    def __len__(self):
        return self.size

    def set_x(self, position):
        self.x = position
    def set_y(self, position):
        self.y = position

class Screen:
    def __init__(self, width, height):
        self.screen = np.zeros((height + 1,width))
        self.width = width
        self.height = height
        self.screen_history = np.array([])

    def resetScreen(self):
        self.screen = np.zeros(self.screen.shape)
        self.screen_history = np.array([])

    def drawAgent(self, agent):
        self.screen[-1,:] = 0
        self.screen[-1,self.wrapper(range(agent.x,agent.x+len(agent)))] = 1

    def drawBlock(self, block):
        self.screen[:-1,:] = 0
        self.screen[block.y,self.wrapper(range(block.x, block.x+len(block)))] = 1

    def drawAgent_cumulative(self, agent):
        self.screen[agent.y,self.wrapper(range(agent.x,agent.x+len(agent)))] = 1

    def drawBlock_cumulative(self, block):
        self.screen[block.y,self.wrapper(range(block.x, block.x+len(block)))] = 1

    def saveCurrentScreen(self):
        if len(self.screen_history)==0:
            self.screen_history = copy.copy(self.screen[np.newaxis,:,:])
        else:
            self.screen_history = np.r_[self.screen_history, self.screen[np.newaxis,:,:]]

    def wrapper(self,index):
        if not hasattr(index, '__len__'):
            return index%self.width
        else:
            return [ix%self.width for ix in index]

class World:
    def __init__(self, width=16, height=35):
        self.width = width # ComplexiPhi world is 35 (and not 34! 34 is the number of updates)
        self.height = height
        self.screen = Screen(self.width, self.height)

    def _runGameTrial(self, trial, agent, block):

        total_time = self.height # 35 time steps, 34 updates
        motor_activity = agent.get_motor_activity(trial)
        
        # t=0 # Initial position (game hasn't started yet.)
        self.screen.resetScreen()
        self.screen.drawAgent(agent)
        self.screen.drawBlock(block)
        self.screen.saveCurrentScreen()

        for t in range(1, total_time):

            agent.x = self.screen.wrapper(agent.x + motor_activity[t])

            if t<total_time:
                if block.direction == 'right':
                    block.x = self.screen.wrapper(block.x + 1)
                else:
                    block.x = self.screen.wrapper(block.x - 1)

                block.y = block.y + 1
            
            self.screen.drawAgent(agent)
            self.screen.drawBlock(block)            
            self.screen.saveCurrentScreen()

        # agent catches the block if it overlaps with it in t=34
        win = self._check_win(block, agent)
        print(win)

        return self.screen.screen_history, win

    def _get_initial_condition(self, trial):
        agent_init_x = trial % self.width
        self.agent.set_x(agent_init_x)

        block_size = self.block_patterns[trial //(self.width * 2)]
        block_direction = 'left' if (trial // self.width) % 2 == 0 else 'right'
        block_value = 'catch' if (trial // (self.width * 2)) % 2 == 0 else 'avoid'
        block = Block(block_size, block_direction, block_value, 0)

        return self.agent, block

    def run_fullgame(self, agent, block_patterns):

        if not hasattr(agent, 'brain_activity'):
            raise AttributeError("Agent needs a brain activity saved to play gameself.")
        self.agent = copy.copy(agent)
        self.block_patterns = block_patterns
        self.n_trials = self.width * 2 * len(block_patterns)

        self.history = np.zeros((self.n_trials,self.height,self.height+1,self.width))

        wins = []
        for trial in range(self.n_trials):
            self.agent, block = self._get_initial_condition(trial)
            self.history[trial,:,:,:], win = self._runGameTrial(trial,self.agent, block)
            wins.append(win)

        self.wins = wins
        return self.history, self.wins

    def get_fullgame_history(self, agent=None, block_patterns=None):
        if hasattr(self, 'history'):
            return self.history
        else:
            self.run_fullgame(agent, block_patterns)
            return self.history

    def _check_win(self, block, agent):
        block_ixs = self.screen.wrapper(range(block.x, block.x + len(block)))
        agent_ixs = self.screen.wrapper(range(agent.x, agent.x + len(agent)))
        catch = True if len(set(block_ixs).intersection(agent_ixs))>0 else False
        win = True if (block.type=='catch' and catch) or (block.type=='avoid' and not catch) else False
        return win

    def get_final_score(self):
        score = 0
        for trial in range(self.n_trials):
            agent, block = self._get_initial_condition(trial)
            # print('trial {}'.format(trial))
            # print('A0: {} B0: {} ({}, {}, {})'.format(agent.x,block.x,len(block),block.direction, block.type))

            agent.x = self.screen.wrapper(agent.x + np.sum(agent.getMotorActivity(trial)[:]))

            direction = -1 if block.direction=='left' else 1
            block.x = self.screen.wrapper(block.x + (self.height-1)*direction)

            win = 'WIN' if self._check_win(block, agent) else 'LOST'
            # print('Af: {} Bf: {}'.format(agent.x, block.x))
            # print(win)
            # print()
            score += int(self._check_win(block, agent))
        print('Score: {}/{}'.format(score, self.n_trials))
        return score
