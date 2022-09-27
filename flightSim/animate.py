import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

FLOOR = -10
CEILING = 10


class AnimatedWithShape(object):
    def __init__(self, num_points=5, shape='scatter'):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.shape = self.init_func(shape)

        self.num_points = num_points
        self.stream = self.data_stream()
        self.angle = 0

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=100, frames=200)

    def change_angle(self):
        self.angle = (self.angle + 1) % 360

    def init_func(self, shape):
        if shape != 'scatter':
            raise NotImplementedError

        x = next(self.stream)
        c = ['b', 'r', 'g', 'y', 'm']
        scat = self.ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c, s=200)

        self.ax.set_xlim3d(FLOOR, CEILING)
        self.ax.set_ylim3d(FLOOR, CEILING)
        self.ax.set_zlim3d(FLOOR, CEILING)

        return scat,

    def data_stream(self):
        data = np.zeros((self.num_points, 3))
        xyz = data[:, :3]
        while True:
            xyz += 2 * (np.random.random((self.num_points, 3)) - 0.5)
            yield data

    def update(self, i):
        data = next(self.stream)
        self.shape._offsets3d = (np.ma.ravel(data[:, 0]), np.ma.ravel(data[:, 1]), np.ma.ravel(data[:, 2]))
        return self.shape,

    def show(self):
        plt.show()


# a = AnimatedScatter()
# a.ani.save('test_animation.gif', writer='Pillow')  # 保存gif...
# a.ani.save('test_animation.mp4') # 保存MP4


def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,  # 千万别少了这里的逗号！！！！！


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot(xdata, ydata, 'r-', animated=True)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1, 1)

ani = animation.FuncAnimation(fig, update, blit=True, fargs=(x, y))
ani.save('test_animation.gif', writer='Pillow')  # 保存 gif
