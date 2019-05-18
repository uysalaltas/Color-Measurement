import time
import numpy as np
import matplotlib.pyplot as plt
import operator
from mpl_toolkits.mplot3d import Axes3D


class ColorMeanCalculator(object):
    color_points = {'Red': [104, 186, 172], 'Orange': [127, 163, 180], 'Green': [117, 87, 172], 'White': [255, 128, 128],
                    'Black': [0, 128, 128], 'Purple': [89, 170, 88], 'Yellow': [214, 113, 205], 'Grey': [102, 128, 128],
                    'Brown': [96, 138, 188], 'Blue': [51, 125, 49]}
    color_lists = ['Red', 'Orange', 'Green', 'White', 'Black', 'Purple', 'Yellow', 'Grey', 'Brown', 'Blue']

    def __init__(self, pixel_array):
        self.pixel_array = pixel_array

    def show_color_space(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        color_array = np.asarray(list(self.color_points.values()))

        X1 = color_array[:, 0]
        Y1 = color_array[:, 1]
        Z1 = color_array[:, 2]

        X2 = self.pixel_array[:, 0]
        Y2 = self.pixel_array[:, 1]
        Z2 = self.pixel_array[:, 2]

        ax.scatter(X1, Y1, Z1, c='b', marker='^')
        ax.scatter(X2, Y2, Z2, c='r', marker='o')

        ax.set_xlim3d(0, 255)
        ax.set_ylim3d(0, 255)
        ax.set_zlim3d(0, 255)

        ax.set_xlabel('A Channel')
        ax.set_ylabel('B Channel')
        ax.set_zlabel('L Channel')

        plt.show()

    def show_pixel_space(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        X = self.pixel_array[:, 0]
        Y = self.pixel_array[:, 1]
        Z = self.pixel_array[:, 2]

        ax.scatter(X, Y, Z)
        ax.set_xlim3d(0, 255)
        ax.set_ylim3d(0, 255)
        ax.set_zlim3d(0, 255)
        ax.set_xlabel('A Channel')
        ax.set_ylabel('B Channel')
        ax.set_zlabel('L Channel')

        plt.show()

    def calculate_distance(self):
        final_colors = []

        color_array = np.asarray(list(self.color_points.values()))
        i = 0
        cluster = []

        for x in self.pixel_array:
            for y in color_array:
                dist = np.linalg.norm(x - y)
                i = i + 1
                cluster.append(dist)

                if i == len(color_array):
                    i = 0
                    min_value = min(cluster)
                    color_code = cluster.index(min_value)
                    cluster.clear()
                    final_colors.append(self.color_lists[color_code])

        return final_colors

    def calculate_colors(self, color_array):

        color_count = {}
        for x in color_array:
            if x not in color_count:
                color_count[x] = 1
            else:
                color_count[x] += 1

        color_count = sorted(color_count.items(), key=operator.itemgetter(1), reverse=True)
        color1 = color_count[0]
        color2 = color_count[1]
        col1 = color1[0]
        col2 = color2[0]

        if int(color2[1]) < 2:
            col2 = None

        return col1, col2

    def calculate_delta_e(self, color_array, code_array, color1, color2=None):
        color1_code = []
        color2_code = []
        de2 = None
        real_c1 = self.color_points.get(color1)

        index_of_color1 = [i for i, value in enumerate(color_array) if value == color1]
        for x in index_of_color1:
            color1_code.append(code_array[x])

        mean_c1 = np.mean(color1_code, axis=0)
        de1 = np.linalg.norm(mean_c1 - real_c1)

        if color2 is not None:
            real_c2 = self.color_points.get(color2)
            index_of_color2 = [index for index, value in enumerate(color_array) if value == color2]
            for x in index_of_color2:
                color2_code.append(code_array[x])
            mean_c2 = np.mean(color2_code, axis=0)
            de2 = np.linalg.norm(mean_c2 - real_c2)

        return de1, de2


# px = np.random.randint(0, 100, size=(300, 3))
# start_time = time.time()
# test = ColorMeanCalculator(px)
# ca = test.calculate_distance()
# c1, c2 = test.calculate_colors(ca)
# a, b = test.calculate_delta_e(ca, px, color1=c1, color2=c2)
#
# print("Time: ", time.time() - start_time)
# print(c1, c2)
# print("Color1 Distance: ", a, " Color2 Distance: ", b)
