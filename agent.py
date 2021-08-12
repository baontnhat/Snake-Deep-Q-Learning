import torch
import random
import numpy as np
from collections import deque
from snake_ai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # parameter to control randomness
        self.gamma = 0.9 # discount rate 
        self.memory = deque(maxlen=MAX_MEMORY) # if exceed auto popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        #to_do: model, trainer

    def get_state(self, game): # state = [11 values]
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger straight 
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l, dir_r, dir_u, dir_d,

            # food location
            game.food.x < game.head.x, # food left to snake
            game.food.x > game.head.x, # food right to snake
            game.food.y < game.head.y, # food above snake
            game.food.y > game.head.y, # food below snake
        ]

        return np.array(state, dtype = int)        

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # exceed memory then popleft()

    def train_long_memory(self): #replay memory or experience replay, train on all previous games
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuple
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    def train_short_memory(self, state, action, reward, next_state, game_over): # train for 1 game step
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation 
        self.epsilon = 80 - self.n_games
        new_action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # smaller epsilon, more randomization then we don't need to randomize anymore
            move = random.randint(0, 2)
            new_action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            new_action[move] = 1
        return new_action
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get current state
        state_old = agent.get_state(game)
        
        # get action
        new_action = agent.get_action(state_old)

        # perform action and get new state
        reward, game_over, score = game.play_step(new_action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, new_action, reward, state_new, game_over)

        # remember
        agent.remember(state_old, new_action, reward, state_new, game_over)

        if game_over:
            #train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score:', score, 'Record:', record)

            #plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
