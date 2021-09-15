from copy import deepcopy

import numpy as np
from utils import *


# -------------- Block World ----------------------
class Block:
    def __init__(self, size, direction, block_type, x_position=0, y_position=0):
        """
		direction: left/right
		block_type: catch/avoid
		"""
        self.size = size
        self.direction = direction
        self.type = block_type
        self.x = x_position
        self.y = y_position

    def __len__(self):
        return self.size


class Game_Agent:
    def __init__(self, agent, x_position=0):
        # agent input can be original agent, or game_agent from previous trial
        self.tpm = agent.tpm
        self.x = x_position
        self.state = np.zeros(len(self.tpm[1]))
        self.pheno_sensor_pos = agent.pheno_sensor_pos
        self.sensor_ixs = agent.sensor_ixs
        self.motor_ixs = agent.motor_ixs
        self.pheno_length = agent.pheno_length
        if hasattr(agent, "current_score"):
            self.current_score = agent.current_score
        else:
            self.current_score = 0

    def reset_state(self):
        self.state = np.zeros(len(self.tpm[1]))

    def update_agent_state(self):
        state_ix = state2num(self.state)
        self.state = deepcopy(self.tpm[state_ix])  # deepcopy here necessary

    def get_motor_activity(self):
        # How the motor state is interpreted in the game
        motor_state = list(
            self.state[self.motor_ixs]
        )  # has to be list for comparison below
        # print('motor_ixs: ', self.motor_ixs)
        motor_activity = 0
        if motor_state == [1, 0]:
            motor_activity = 1
        elif motor_state == [0, 1]:
            motor_activity = -1
        return motor_activity


class Block_World:
    def __init__(self, width=16, height=34, block_patterns=[3, 4, 6, 5]):
        self.width = (
            width  # ComplexiPhi world is 35 (and not 34! 34 is the number of updates)
        )
        self.height = height
        self.block_patterns = block_patterns
        self.num_trials = len(block_patterns) * width * 2

    def wrapper(self, index):
        if not hasattr(index, "__len__"):
            return index % self.width
        else:
            return [ix % self.width for ix in index]

    def check_hit(self, game_agent, block):
        agent_position = self.wrapper(
            range(game_agent.x, game_agent.x + game_agent.pheno_length)
        )

        world_section = np.zeros(self.width)
        block_pos = self.wrapper(range(block.x, block.x + block.size))
        world_section[block_pos] = 1
        # print('final world: ', world_section)
        # print('final_agent: ', agent_position)
        overlap = sum(world_section[agent_position])

        if overlap > 0:
            hit = True
        else:
            hit = False

        return hit

    def _update_agent_position(self, game_agent, world_section):
        agent_rel_sensor_position = self.wrapper(
            [game_agent.x + s for s in game_agent.pheno_sensor_pos]
        )
        agent_sensor_state = world_section[agent_rel_sensor_position]

        game_agent.state[game_agent.sensor_ixs] = agent_sensor_state
        # print(game_agent.state)

        game_agent.update_agent_state()
        # print(game_agent.state)

        game_agent.x = self.wrapper(game_agent.x + game_agent.get_motor_activity())
        # print(game_agent.x)

    def _get_initial_conditions_from_trial_num(self, trial, agent):
        # for all block sizes, for [left, right], for initial conditions
        agent_ini_x = trial % self.width

        game_agent = Game_Agent(agent, x_position=agent_ini_x)

        block_size = self.block_patterns[trial // (self.width * 2)]
        block_direction = "left" if (trial // self.width) % 2 == 0 else "right"
        block_type = "catch" if (trial // (self.width * 2)) % 2 == 0 else "avoid"

        block = Block(block_size, block_direction, block_type, 0)

        return game_agent, block

    def _runTrial(self, trial, agent):
        total_time = self.height
        game_agent, block = self._get_initial_conditions_from_trial_num(trial, agent)

        game_agent.reset_state()

        hit = False
        win = False

        # print(['game_agent: ', game_agent.x])

        while block.y < total_time:

            world_section = np.zeros(self.width)
            block_pos = self.wrapper(range(block.x, block.x + block.size))
            world_section[block_pos] = 1
            # print(['block: ', block.x])
            # print(world_section)
            # print('t: ', block.y)
            self._update_agent_position(game_agent, world_section)

            # if block.y == total_time - 1:
            # 	break

            if block.direction == "left":
                block.x = self.wrapper(block.x - 1)
            elif block.direction == "right":
                block.x = self.wrapper(block.x + 1)

            block.y += 1

        hit = self.check_hit(game_agent, block)

        if (block.type == "catch" and hit == True) or (
            block.type == "avoid" and hit == False
        ):
            game_agent.current_score += 1
            win = True

        # print(['hit: ', hit, 'win: ', win, 'block: ', block.x, 'agent: ', game_agent.x])

        return game_agent, win

    def run_full_game(self, agent):

        game_agent = Game_Agent(agent)

        for t in range(self.num_trials):
            game_agent, win = self._runTrial(t, game_agent)
            print([t, game_agent.current_score])

        return game_agent.current_score / self.num_trials
