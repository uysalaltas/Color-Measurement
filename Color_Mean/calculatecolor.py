import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator
from mpl_toolkits.mplot3d import Axes3D


class ColorMeanCalculator(object):
    color_points = {'Red': [41, 58, 44], 'Orange': [50, 35, 52], 'Green': [46, -41, 44], 'White': [100, 0, 0],
                    'Black': [0, 0, 0], 'Purple': [35, 42, -40], 'Yellow': [84, -15, 77], 'Grey': [40, 0, 0],
                    'Brown': [38, 10, 60], 'Blue': [20, -3, -79]}

    color_lists = ['Red', 'Orange', 'Green', 'White', 'Black', 'Purple', 'Yellow', 'Grey', 'Brown', 'Blue']

    def __init__(self, pixel_array):
        self.pixel_array = pixel_array

    def show_color_space(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        values = [x for x in self.color_points.values()]
        color_array = np.asarray(values)

        X = color_array[:, 0]
        Y = color_array[:, 1]
        Z = color_array[:, 2]

        ax.scatter(X, Y, Z)
        ax.set_xlabel('A Channel')
        ax.set_ylabel('B Channel')
        ax.set_zlabel('L Channel')

        plt.show()

    def calculate_distance(self):
        final_colors = []

        values = [x for x in self.color_points.values()]
        color_array = np.asarray(values)
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
        color_count = {'Red': 0, 'Orange': 0, 'Green': 0, 'White': 0, 'Black': 0, 'Purple': 0, 'Yellow': 0, 'Grey': 0,
                       'Brown': 0, 'Blue': 0}

        for x in color_array:
            if x == 'Red':
                color_count['Red'] += 1
            if x == 'Orange':
                color_count['Orange'] += 1
            if x == 'Green':
                color_count['Green'] += 1
            if x == 'White':
                color_count['White'] += 1
            if x == 'Black':
                color_count['Black'] += 1
            if x == 'Purple':
                color_count['Purple'] += 1
            if x == 'Yellow':
                color_count['Yellow'] += 1
            if x == 'Grey':
                color_count['Grey'] += 1
            if x == 'Brown':
                color_count['Brown'] += 1
            if x == 'Blue':
                color_count['Blue'] += 1

        color_count = sorted(color_count.items(), key=operator.itemgetter(1), reverse=True)
        print(color_count)
        color1 = color_count[0]
        color2 = color_count[1]
        col1 = color1[0]
        col2 = color2[0]

        if int(color2[1]) < 2:
            print("Color 2 not exist")
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


px = np.random.randint(0, 100, size=(300, 3))
start_time = time.time()
test = ColorMeanCalculator(px)
ca = test.calculate_distance()
c1, c2 = test.calculate_colors(ca)
a, b = test.calculate_delta_e(ca, px, color1=c1, color2=c2)

print("Time: ", time.time() - start_time)
print(c1, c2)
print("Color1 Distance: ", a, " Color2 Distance: ", b)
