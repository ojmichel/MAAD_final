import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import argparse
import matplotlib.animation as animation

class Grid(object):

    WIDTH = 50
    HEIGHT = 50
    DIMENSION = 3

    def __init__(self,parser):

        args = parser.parse_args()
        self.A = 100.0
        self.R = 100.0
        self.t = 0
        self.sigma = 0.1
        self.mode = -1.0
        self.frames = []
        self.rad = 1.0
        self.save = False
        self.video_fp = 'video.mp4'

        if args.W:
            self.WIDTH = args.W
        if args.H:
            self.HEIGHT = args.H
        if args.mode:
            if args.mode == 'mix':
                self.mode=1.0
            elif args.mode == 'bar':
                self.mode = -1.0
        if args.s:
            self.save = True
            self.video_fp = args.s
        if args.sigma:
            self.sigma = args.sigma
        if args.rad:
            self.rad = args.rad


        if not args.f:
            if args.init == 'rand':
                self.grid = np.random.normal(scale=args.sigma_init,size=(self.WIDTH,self.HEIGHT,self.DIMENSION))
            else:
                self.grid = np.zeros((self.WIDTH,self.HEIGHT,self.DIMENSION))
        else:
            img = np.array(Image.open(args.f))
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

    def update(self):

        frame = np.copy(self.grid)
        frame = ((frame + 1.0)/2.0)*255.0
        frame = frame.astype('uint8')
        self.frames.append(frame)

        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                self.grid[x,y] += self.attract(x,y,mix=self.mode) + self.repel(x,y) + self.random()
                self.embed(x,y)
        self.t += 1


        return self.grid

    def get_neighbors(self,x,y):
        return np.array([self.grid[i,j] for i,j in self.neighbors[x,y]])

    def attract(self,x,y,mix=1.0): #-1.0 get bar mode
        v = self.grid[x,y]
        n = self.get_neighbors(x,y)
        norm = np.linalg.norm(n - v,axis=1)
        e = self.A*np.exp(mix*norm)
        delta = np.sum((e * (n-v).T).T,axis=0)
        return delta / len(n)

    def repel(self,x,y):

        v = self.grid[x, y]
        n = self.get_neighbors(x, y)
        f = 0.0
        norm = np.linalg.norm(n - v, axis=1)
        t = self.R * np.tanh(norm - f)
        delta = np.sum((t * (v - n).T).T, axis=0)
        return delta / len(n)



    def random(self):
        return np.random.normal(scale=self.sigma,size=self.DIMENSION)

    def embed(self, x, y):
        norm = np.linalg.norm(self.grid[x,y])
        if norm > self.rad:
            self.grid[x,y] = self.rad * self.grid[x,y] / norm

    def plot1(self,data):
        data = (data + self.rad)/(2.0*self.rad)
        self.p1.clear()
        self.p1.matshow(data)
    def plot2(self,data):
        c = (data + self.rad)/(2.0*self.rad)
        c = c.reshape((self.WIDTH*self.HEIGHT,3))
        self.p2.clear()
        self.p2.scatter(data[:,:,0],data[:,:,1],data[:,:,2],c=c)
    def gen(self):

        while True:
            self.update()
            yield self.grid



    def save_vid(self,x=None):
        if self.save:
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            writer = cv2.VideoWriter(self.video_fp,fourcc,30,(600,600))
            for f in self.frames:
                writer.write(cv2.resize(f,(600,600)))
            writer.release()




    def run(self):
        self.fig = plt.figure(figsize=(8,8))
        self.fig.canvas.mpl_connect('close_event',self.save_vid)
        self.p1 = self.fig.add_subplot(2, 1, 1)
        self.p2 = self.fig.add_subplot(2, 1, 2,projection='3d')
        self.ani1 = animation.FuncAnimation(self.fig, self.plot1, self.gen,interval=1)
        self.ani2 = animation.FuncAnimation(self.fig, self.plot2, self.gen,interval=1)
        plt.show()



if __name__ == '__main__':
    # Grid(fp='image100.jpeg').run()

    parser = argparse.ArgumentParser()
    parser.add_argument('--f',help='input file',type=str)
    parser.add_argument('--init',help='zero or rand',type=str,default='zero')
    parser.add_argument('--s',help='file to save animation',type=str)
    parser.add_argument('--H',help='height',type=int)
    parser.add_argument('--W', help='width',type=int)
    parser.add_argument('--mode',help='mix or bar mode',type=str)
    parser.add_argument('--sigma',help='std of random noise',type=float)
    parser.add_argument('--sigma_init',type=float,default=1.0)
    parser.add_argument('--rad',help='radius',type=float)

    Grid(parser).run()

