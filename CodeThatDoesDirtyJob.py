#https://github.com/bdhammel/least-squares-ellipse-fitting
import random
import Measurments
import numpy as np
from ellipse import LsqEllipse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class PerformMeasurments:

    ACCURACY = 3000
    from_np = False

    def img_to_pairs(self, img = None, drop_prob = 0, channel = 0):

        if(not self.from_np):
            img = Image.open(img)
            img_data = np.asarray(img)
        else:
            img_data = self.seg_np

        img_data = img_data[:,:,channel]
        img_data_edge = cv2.Canny(img_data, 1000, 200)
        channels = img_data_edge

        data = np.asarray(channels)
        print("Channels seperated" , data.shape)
        X,Y = data.shape
        points_x = []
        points_y = []

        for x in range(X):
            for y in range(Y):
                if data[x][y] != 0:
                    if random.randint(0, drop_prob) == 0:
                        points_x.append(x + 0.0)
                        points_y.append(y + 0.0)
                    else:
                        pass
                        #print("Skipped")
        return [np.array(points_x), np.array(points_y)]

    def endpoint_coordinates(self,center, width, height, phi):
        acs = width * np.cos(phi)
        asn = width * np.sin(phi)
        x1 =  center[0] + acs
        y1 =  center[1] + asn
        x2 =  center[0] - acs
        y2 =  center[1] - asn
        return x1,y1,x2,y2
    def endpoint_coordinates_on_width(self,center, width, height, phi):
        # Calculate the components for the width
        wcs = width * np.cos(phi)
        wsn = width * np.sin(phi)

        # Calculate the coordinates of the endpoints along the width
        x1 = center[0] + wcs
        y1 = center[1] + wsn
        x2 = center[0] - wcs
        y2 = center[1] - wsn

        return x1, y1, x2, y2

    def ellipse_y(self,ellipse,x, above = 2, accuracy = 1000):
        t = np.linspace(0, 2 * 3.14, accuracy)
        XY_pairs = ellipse.return_fit(t=t)

        mask = XY_pairs[:,1] > above
        X = XY_pairs[: , 0][mask]
        Y = XY_pairs[: , 1][mask]

        idx = (np.abs(X - x)).argmin()
        return X[idx] , X[idx + 1] , Y[idx] , Y[idx + 1]

    def compute_distances(self,x, y, x0, y0):
        distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        return distances

    def __init__(self, filename):
        self.filename = filename
    def visualize(self):
        X1_small, X2_small = self.img_to_pairs(self.filename, 0, 0)
        X1_big, X2_big = self.img_to_pairs(self.filename, 1, 1)

        X1_small /= 100
        X2_small /= 100

        X1_big /= 100
        X2_big /= 100

        X_small = np.array(list(zip(X1_small, X2_small)))
        X_big = np.array(list(zip(X1_big, X2_big)))

        reg_small = LsqEllipse().fit(X_small)
        reg_big = LsqEllipse().fit(X_big)

        center_s, width_s, height_s, phi_s = reg_small.as_parameters()
        center_b, width_b, height_b, phi_b = reg_big.as_parameters()

        # Get endpoint coordinates
        x1, y1, x2, y2 = self.endpoint_coordinates(center_s, width_s, height_s, phi_s)
        x1_b, y1_b, x2_b, y2_b = self.endpoint_coordinates_on_width(center_b, width_b, height_b, phi_b)

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot()

        ax.axis('equal')
        # ax.plot(X1_small, X2_small, 'ro', zorder=1)
        # ax.plot(X1_big, X2_big, 'ro', zorder=1)

        ellipse = Ellipse(
            xy=center_b, width=2 * width_b, height=2 * height_b, angle=np.rad2deg(phi_b),
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )
        ellipse_b = Ellipse(
            xy=center_s, width=2 * width_s, height=2 * height_s, angle=np.rad2deg(phi_s),
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
        )

        ax.add_patch(ellipse)
        ax.add_patch(ellipse_b)

        msr = Measurments.Measurments()
        pivot_point = [x1, y1]

        # Find the closest point in big and for x1 and y1
        t = np.linspace(0, 2 * 3.14, self.ACCURACY)
        XY_pairs = reg_big.return_fit(t=t)
        closest_point = msr.find_closest(pivot_point, XY_pairs)

        Vector1 = msr.get_vector(pivot_point, [x2, y2])
        Vector2 = msr.get_vector(pivot_point, closest_point)

        AngleSmall = msr.get_angle(Vector1, Vector2)
        print("Small angle (HSD) : ", AngleSmall)

        # Visualizing angle ADD later
        ax.plot([x2, x1], [y2, y1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        ax.plot([x1, closest_point[0]], [y1, closest_point[1]], marker="o", markersize=5, markeredgecolor="red",
                markerfacecolor="green")

        ###
        ### Computing big angle
        ###

        x_tangent, y_tangent = msr.find_farthest(pivot_point, XY_pairs, 0.1, mini=closest_point[1])
        point_on_head = [x_tangent, y_tangent]

        # Calculate the vector and angle (assuming you have the functions for getting vectors and angles)
        Vector3 = msr.get_vector(pivot_point, point_on_head)
        AngleBig = msr.get_angle(Vector1, Vector3)
        print("Big angle (AoP):", AngleBig)

        # Plotting the tangent line from the point to the tangent point on the ellipse
        ax.plot([x1, point_on_head[0]], [y1, point_on_head[1]], 'r-')

        plt.text(pivot_point[0], pivot_point[1], f'(HSD) = {AngleSmall:.2f}', fontsize=12)
        plt.text(pivot_point[0] + 1, pivot_point[1], f'(AoP) = {AngleBig:.2f}', fontsize=12)

        """
        Plotting final picture
        """
        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$')

        plt.legend()
        plt.show()


if __name__ == "__main__":
    PM = PerformMeasurments("example3.png")
    PM.visualize()