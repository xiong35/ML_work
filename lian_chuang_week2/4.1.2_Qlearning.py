
import numpy as np
import os
from time import sleep


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
        self.alpha = 0.25
        self.gamma = 0.99
        self.epsilon = 1.
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

    def train(self):
        my_map = Map()
        my_map.plot(tuple(self.position))
        count = 0
        while True:
            while self.position not in [[1, 3], [3, 2], [3, 4], [4, 0]]:
                count += 1
                [x, y] = self.position
                q_vals = self.q_table[x, y, :]
                if (np.random.uniform(0, 1) < self.epsilon) or (q_vals == 0).all():
                    movement = np.random.randint(0, 4)
                else:
                    movement = int(np.where(q_vals == q_vals.max())[0][0])

                [x_next, y_next] = self.move(movement)
                next_q = self.q_table[x_next, y_next, :].max()
                self.q_table[x, y, movement] += self.alpha * (
                    my_map.map[x_next, y_next]+self.gamma*next_q
                    - self.q_table[x, y, movement])
                self.position = [x_next, y_next]
                if count > 1200:
                    my_map.plot((x_next, y_next), 0.15)
                else:
                    my_map.plot((x_next, y_next))
                if self.epsilon > 0.01:
                    self.epsilon *= 0.995
            self.position = [0, 0]


ag = Agent()
ag.train()
