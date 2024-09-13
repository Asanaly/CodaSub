import numpy as np


class Measurments:
    """
    Util class
    """
    def __self__(self, endpoint_left , endpoint_right):
        pass

    def set_end(self , endpoint_left , endpoint_right):
        self.endpoint_left = np.asarray(endpoint_left)
        self.endpoint_right = np.asarray(endpoint_right)

    def get_vector(self, point1 , point2):
        #Returns vector from vector1 to vector2
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        tangent_vector = point2 - point1
        distance = self.dist(point2, point1)
        unit_tan_vector_end = tangent_vector / distance
        return unit_tan_vector_end

    def dist(self, point1 , point2):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        return pow( pow(x2 - x1 , 2) + pow(y2 - y1 , 2) , 0.5 )

    def get_angle(self, vec1 , vec2):
        #print(vec1, vec2)
        #Dot = |a||b|cos(delta)
        angle = np.arccos( np.dot(vec1,vec2) ) * 180 / np.pi
        return angle

    def find_closest(self, stat_point , points):
        """Returns closest point in points to the stat_point"""
        stat_point = np.asarray(stat_point)
        points = np.asarray(points)
        quadro = ((points - stat_point) ** 2)
        dist = [np.sqrt(sum(point)) for point in quadro]
        dist = np.asarray(dist)
        #print(dist)
        closest = dist.argmin()
        #print(closest)
        return points[closest]

    def is_point_on_line(self,A, B, C, tolerance=1e-2, pr = 0):
        #Returns 1 if ABC are on the same line
        if C[0] > B[0]:
            return 0


        x1, y1 = A
        x2, y2 = B
        x3, y3 = C

        #BC
        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y3 - y2) / (x3 - x2)

        return abs(slope1 - slope2) < tolerance



    def find_farthest(self, stat_point, points, tolerance = 1e-2, mini = 0):
        stat_point = np.asarray(stat_point)
        points = np.asarray(points)

        max_distance = -1
        tangent_line = np.zeros([2])

        x = points[:,0]
        y = points[:,1]

        for i,_ in enumerate(x):
            if y[i] < mini:
                continue
            point = [x[i] , y[i]]
            point_between = 0
            if self.dist(stat_point,point) < max_distance:
                continue

            for j,_ in enumerate(x):
                C = [x[j] , y[j]]
                if i == j:
                    continue

                point_between = self.is_point_on_line(stat_point,point,C, tolerance=tolerance)

                if point_between:
                    break

            #If point_between 0 there is no points in between
            if not point_between:
                distance = self.dist(stat_point, point)
                if distance > max_distance:
                    max_distance = distance
                    tangent_line = np.asarray(point)

        return tangent_line




