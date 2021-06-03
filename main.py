import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import matplotlib.animation as animation

class Grid(object):

    WIDTH = 100
    HEIGHT = 100
    DIMENSION = 3

    def __init__(self,fp=None):

        if not fp:
            # self.grid = np.random.normal(size=(self.WIDTH,self.HEIGHT,self.DIMENSION))
            self.grid = np.zeros((self.WIDTH,self.HEIGHT,self.DIMENSION))
        else:
            img = np.array(Image.open(fp))
            img = cv2.resize(img,(self.WIDTH,self.HEIGHT))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            self.grid = np.array(img,dtype='float64')

        self.neighbors = {}
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                n = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        # good = x + dx < self.WIDTH and y + dy < self.HEIGHT and (not (dx == 0 and dy == 0))
                        good = (not (dx == 0 and dy == 0))
                        if good:
                            n.append(((x + dx) % self.WIDTH, (y + dy) % self.HEIGHT))
                            # n.append((x + dx,y + dy))
                self.neighbors[x,y] = n


        self.A = 100.0
        self.R = 100.0
        self.sigma = 10.0




    def update(self):
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                self.grid[x,y] += self.attract(x,y) + self.repel(x,y) + self.random()
                self.embed(x,y)
        self.display_grid = (self.grid + 1.0)* (255.0 / 2.0)
        self.display_grid.dtype='int64'
        return self.grid

    def get_neighbors(self,x,y):
        return np.array([self.grid[i,j] for i,j in self.neighbors[x,y]])

    def bars(self,x,y):
        neighbors = self.get_neighbors(x, y)
        v = self.grid[x, y]
        delta = np.zeros(3)
        for n in neighbors:
            delta += self.A * np.exp(-np.linalg.norm(n - v)) * (n - v)
        # print(delta / len(neighbors))
        # v = self.grid[x,y]
        # n = self.get_neighbors(x,y)
        # norm = np.linalg.norm(n - v,axis=1)
        # e = self.A*np.exp(norm)
        # delta = np.sum(e)*(n - v)
        return delta / len(neighbors)

    def attract(self,x,y):
        v = self.grid[x,y]
        n = self.get_neighbors(x,y)
        norm = np.linalg.norm(n - v,axis=1)
        e = self.A*np.exp(norm)
        delta = np.sum((e * (n-v).T).T,axis=0)
        return delta / len(n)

    def repel(self,x,y):
        # neighbors = self.get_neighbors(x, y)
        # v = self.grid[x, y]
        # f=0.0
        # delta = np.zeros(3)
        # for n in neighbors:
        #     delta += self.R*np.tanh(np.linalg.norm(n - v) - f)*(v - n)

        # print(delta/len(neighbors))
        v = self.grid[x, y]
        n = self.get_neighbors(x, y)
        f = 0.0
        norm = np.linalg.norm(n - v, axis=1)
        t = self.R * np.tanh(norm - f)
        delta = np.sum((t * (n - v).T).T, axis=0)
        return delta / len(n)

    def bump(self,x,y):
        neighbors = self.get_neighbors(x, y)
        v = self.grid[x, y]
        delta = np.zeros(3)
        for n in neighbors:
            delta += self.R * np.exp(1.0 / (1/(np.linalg.norm(n - v)**2 - 1))) * (v - n)
        # print(delta/len(neighbors))
        return delta / len(neighbors)


    def random(self):
        return np.random.normal(scale=self.sigma,size=self.DIMENSION)

    def embed(self, x, y):
        norm = np.linalg.norm(self.grid[x,y])
        if norm > 1.0:
            self.grid[x,y] = self.grid[x,y] / norm

    def plot1(self,data):
        self.p1.clear()
        self.p1.matshow(data)
    def plot2(self,data):
        self.p2.clear()
        self.p2.scatter(10.0*data[:,:,0],10.0*data[:,:,1],10.0*data[:,:,2],c=data.reshape((self.WIDTH*self.HEIGHT,3)))
    def gen(self):
        while True:
            self.update()
            yield (self.grid + 1.0)/2.0

    def on_close(self,x):
        writer = animation.PillowWriter()
        self.ani1.save('ani1.gif',writer)


    def run(self):
        self.fig = plt.figure(figsize=(8,8))
        # self.fig.canvas.mpl_connect('close_event',self.on_close)
        self.p1 = self.fig.add_subplot(2, 1, 1)
        self.p2 = self.fig.add_subplot(2, 1, 2,projection='3d')
        self.ani1 = animation.FuncAnimation(self.fig, self.plot1, self.gen, interval=1000)
        self.ani2 = animation.FuncAnimation(self.fig, self.plot2, self.gen, interval=1000)
        plt.show()



if __name__ == '__main__':
    Grid(fp='salty.jpeg').run()
    # Grid().run()

