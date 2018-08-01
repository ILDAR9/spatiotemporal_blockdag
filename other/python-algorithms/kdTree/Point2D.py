import math

X_COORD = 0
Y_COORD = 1

class Point2D:
    """
    A point with 2 coordinates, X_COORD and Y_COORD
    """

    def __init__(self, xCoord, yCoord):
        """
        The coordinates of the point.

        :type xCoord: float
        :param xCoord: the X coordinate of the point
        :type yCoord: float
        :param yCoord: the Y coordinate of the point
        """
        if math.isnan(xCoord) or math.isnan(yCoord): raise ValueError("Coordinates cannot be NaN!")
        if math.isinf(xCoord) or math.isinf(yCoord): raise ValueError("Coordinates cannot be Inf!")
        self.coords = [xCoord, yCoord]

    def __eq__(self, other):
        return self.coords == other.coords

    def __ne__(self, other):
        return self.coords != other.coords

    def squaredDistanceTo(self, other):
        """
        The squared distance from this point to the other given point.

        :type other: Point2D
        :param other: another point to calculate the squared distance to.
        :return: the squared distance from this point and the other point given
        """
        sq = 0
        for idx, coord in enumerate(self.coords):
            dc = coord - other.coords[idx]
            sq += dc * dc

        return sq


    def distanceTo(self, other):
        """
        The distance from this point to the other given point.

        :type other: Point2D
        :param other: another point to calculate the distance to.
        :return: the distance from this point and the other point given
        """
        return math.sqrt(self.squaredDistanceTo(other))


