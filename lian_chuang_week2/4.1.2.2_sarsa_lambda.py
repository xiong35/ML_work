
import numpy as np
import os
from time import sleep
np.random.seed(77)

class Map:
    def __init__(self):
        self.size = 5
        self.map = np.zeros((self.size, self.size))
        self.map[1, 3] = -100
        self.map[3, 2] = -100
        self.map[3, 4] = 100
        self.map[4, 0] = -100

    def plot(self, hero_position, interval=0):
        os.system('cls')
        cur_map = self.map.copy()
        cur_map[hero_position] = 96
        print(cur_map)
        sleep(interval)


class Agent:
    def __init__(self):
        self.alpha = 0.25   # lr
        self.gamma = 0.9    # how much futher actions influence NOW
        self.epsilon = 1.   # gready
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.trace_decay = 0.8
        self.position = [0, 0]
        self.q_table = np.zeros((5, 5, 4))  # 5*5 map / 4 direcrions

    def move(self, movement):
        next_pos = self.position
        if movement == 0 and self.position[0] != 0:
            next_pos[0] = self.position[0] - 1
        elif movement == 1 and self.position[0] != 4:
            next_pos[0] = self.position[0] + 1
        elif movement == 2 and self.position[1] != 0:
            next_pos[1] = self.position[1] - 1
        elif movement == 3 and self.position[1] != 4:
            next_pos[1] = self.position[1] + 1
        return next_pos

    def next_move(self, q_vals):
        rand = np.random.uniform(0, 1)
        if rand < self.epsilon or (q_vals == 0).all():
            movement = np.random.randint(0, 4)
        else:
            movement = np.argmax(q_vals)
        return movement

    def train(self):
        my_map = Map()
        my_map.plot(tuple(self.position))
        count = 0
        while True:
            e_trace = np.zeros((5, 5, 4))
            self.position = [0, 0]
            movement = self.next_move(self.q_table[0, 0, :])
            while self.position not in [[1, 3], [3, 2], [3, 4], [4, 0]]:
                count += 1
                [x, y] = self.position

                [x_next, y_next] = self.move(movement)
                if [x_next, y_next] == [x, y]:
                    self.q_table[x, y, movement]-=100
                next_q_vals = self.q_table[x_next, y_next, :]
                next_movement = self.next_move(next_q_vals)

                reward = my_map.map[x_next, y_next]

                delta = reward + self.gamma * \
                    self.q_table[x_next, y_next, next_movement] - \
                    self.q_table[x, y, movement]

                e_trace[x, y, movement] += 1
                self.q_table += self.alpha * delta * e_trace

                e_trace = self.gamma * self.trace_decay * e_trace

                if count > 1200:
                    my_map.plot((x_next, y_next), 0.15)
                else:
                    my_map.plot((x_next, y_next))
                if self.epsilon > self.min_epsilon:
                    self.epsilon *= self.epsilon_decay
                
                self.position = [x_next, y_next]
                movement = next_movement


ag = Agent()
ag.train()
