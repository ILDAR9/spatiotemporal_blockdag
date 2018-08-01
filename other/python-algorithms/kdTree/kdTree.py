import math

X_COORD = 0
Y_COORD = 1
Z_COORD = 2


class Point:
    """A point with K coordinates.

    While the point can have the desired number of dimensions, the first three are usually called
    X_COORD, Y_COORD and Z_COORD and they are at position 0, 1 and 2 of the coordinate list.
    """

    def __init__(self, coordinates):
        """A K dimensional point from a list with K dimensions.

        :type coordinates: list
        :param coordinates: a list with all the coordinates for the point
        """
        for coord in coordinates:
            if math.isnan(coord):
                raise ValueError("Coordinates cannot be NaN!")
            if math.isinf(coord):
                raise ValueError("Coordinates cannot be Inf!")

        self.coords = coordinates[:]

    def __eq__(self, other):
        return self.coords == other.coords

    def __ne__(self, other):
        return self.coords != other.coords

    def dimensions(self):
        return len(self.coords)

    def squared_distance_to(self, other):
        """The squared distance from this point to the other given point.

        :type other: Point
        :param other: another point to calculate the squared distance to.
        :return: the squared distance from this point and the other point given
        """
        sq = 0
        for idx, coord in enumerate(self.coords):
            dc = coord - other.coords[idx]
            sq += dc * dc

        return sq

    def distance_to(self, other):
        """The distance from this point to the other given point.

        :type other: Point
        :param other: another point to calculate the distance to.
        :return: the distance from this point and the other point given
        """
        return math.sqrt(self.squared_distance_to(other))


class Node:
    """A node of the K dimensional tree.
    """
    def __init__(self, point, recHV=None):
        """Create a node with the give point.

        :type point: Point
        :param point: The point for this node.
        """
        self.point = point
        self.recHV = recHV
        self.left = None
        self.right = None


class KdTree:
    """A set of points, internally represented as a kdTree for fast proximity search.

    When building a KdTree set the number of dimensions to use for ordering should be provided if the default (2)
    is not desired. All the entered point then need to provide that required minimum of coordinates.
    Distance is compared on all available dimensions, regardless of how many dimensions are used in ordering.
    """


    def __init__(self, dimensions=2):
        """Constructs a K dimension Tree.

            The default number of dimensions used is 2.

            :rtype: KdTree
            """
        self._root = None
        self._count = 0
        self._dim = dimensions

    def is_empty(self):
        return self._root is None

    def size(self):
        return self._count

    def insert(self, point):
        """Add the point to the set (if it is not already in the set)

        :param point: the point to add to the set
        :type point: Point
        """
        self._root = self._insert(self._root, point, 0)

    def _insert(self, node, point, level):
        if node is None:
            self._count += 1
            node = Node(point)
            return node

        ldist = self.leveled_distance(node, point, level)

        if ldist < 0:
            node.left = self._insert(node.left, point, level + 1)
        elif ldist != 0 or node.point != point:
            node.right = self._insert(node.right, point, level + 1)
        return node

    def _dim_by_level(self, level):
        return level % self._dim

    def leveled_distance(self, node, point, level):
        """Compares two points (the node's point and the given point) on a coordinate based on the node level,
         reporting if the point coordinate is smaller, equal or bigger than the node's point coordinate.

        :param node: the current node
        :type node: Node
        :param point: the point to add
        :type point: Point
        :param level: the level of the node
        :type level: int
        :return: a negative integer, zero, or a positive integer as the point coordinate is
        less than, equal to, or greater than the node's point coordinate determined by the level.
        :rtype: int
        """
        dim = self._dim_by_level(level)
        return point.coords[dim] - node.point.coords[dim]

    def contains(self, point):
        return self.search(point) is not None

    def search(self, point):
        """Searches for a point in the set.

        :param point: the point to look for
        :type point: Point
        :return: the point, if found, or None, if not found.
        :rtype: Point
        """
        node = self._search_node(point, self._root, 0)
        if node is not None:
            return node.point
        else:
            return None

    def _search_node(self, point, node, level):
        if node is None:
            return None
        elif node.point == point:
            return node

        ldist = self.leveled_distance(node, point, level)

        if ldist < 0:
            found = self._search_node(point, node.left, level + 1)
        else:
            found = self._search_node(point, node.right, level + 1)

        return found

    def nearest(self, point):
        """The nearest neighbor in the set to the given point; None if the set is empty.

        :param point: the point we want to find the nearest neigbour in the set
        :type point: Point
        :return: the nearest point found in the set or None if the set is empty.
        :rtype: Point
        """
        if self.is_empty():
            return None

        nnode = self._nearest_node(point, self._root, 0)
        return nnode.point

    def _nearest_node(self, point, node, level):
        if node is None:
            return None

        # Check most promising side
        ldist = self.leveled_distance(node, point, level)
        if ldist < 0:
            nearest = self._nearest_node(point, node.left, level + 1)
        else:
            nearest = self._nearest_node(point, node.right, level + 1)

        # Update nearest from promising side
        if nearest is None:
            nearest = node
            sqrdnrst = point.squared_distance_to(node.point)
        else:
            sqrd = point.squared_distance_to(node.point)
            sqrdnrst = point.squared_distance_to(nearest.point)
            if sqrd < sqrdnrst:
                sqrdnrst = sqrd
                nearest = node

        # Evaluate if less promising side must be searched too
        sqrdldist = ldist * ldist
        if sqrdnrst > sqrdldist:
            if ldist < 0:
                nearest2 = self._nearest_node(point, node.right, level + 1)
            else:
                nearest2 = self._nearest_node(point, node.left, level + 1)

            if nearest2 is not None:
                sqrdnrst2 = point.squared_distance_to(nearest2.point)
                if sqrdnrst2 < sqrdnrst:
                    # sqrdnrst = sqrdnrst2  # currently not used
                    nearest = nearest2

        # Return updated nearest from searching both sides, if required.
        return nearest

    def range(self, min_point, max_point ):
        """Lists the points in the set included in the range defined by the two given points.

        The min_point must have the lower bound for every coordinate, while the max point must have the higher bound.

        :param min_point: the point with coordinates equal to the lowest bounds of the desired range.
        :type min_point: Point
        :param max_point: the point with coordinates equal to the highest bounds of the desired range.
        :type max_point: Point
        :return: a list with the points of the set falling in the given range.
        :rtype: list
        """
        return self._range(self._root, 0, min_point, max_point)

    def _range(self, node, level, min_point, max_point):
        if node is None:
            return []

        points = []

        # Check left side of the tree, if needed
        min_dist = self.leveled_distance(node, min_point, level)
        if min_dist < 0:
            points.extend(self._range(node.left, level + 1, min_point, max_point))

        # Check right side of the tree, if needed
        max_dist = self.leveled_distance(node, max_point, level)
        if max_dist >= 0:
            points.extend(self._range(node.right, level + 1, min_point, max_point))

        # Return the points found, if the current node's point is NOT inside the range
        for i, coord in enumerate(node.point.coords):
            if not min_point.coords[i] <= coord <= max_point.coords[i]:
                return points

        points.append(node.point)
        return points
