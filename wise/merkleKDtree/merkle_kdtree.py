#!/usr/bin/python
# encoding: utf-8
""" KDTree implementation.

Features:

- nearest neighbours search
- range search
"""
import hashlib as hasher

def square_distance(pointA, pointB):
    # squared euclidean distance
    # dimensions = len(pointA) # assumes both points have the same dimensions
    # for dimension in range(dimensions):
    distance = (pointA[0] - pointB[0]) ** 2
    distance += (pointA[1] - pointB[1]) ** 2
    distance += (pointA[2] - pointB[2]) ** 2
    return distance # no need square root them


class KDTreeNode():

    def __init__(self, point, left, right):
        self.point_hash = hasher.sha256("".join((str(x) for x in point)).encode('utf-8')).hexdigest()
        self.point = tuple(point[:3])
        self.left = left
        self.right = right

    def is_leaf(self):
        return (self.left == None and self.right == None)


class KDTreeNeighbours():
    """ Internal structure used in nearest-neighbours search.
    """
    def __init__(self, query_point, t):
        self.query_point = query_point
        self.t = t # neighbours wanted
        self.largest_distance = 0 # squared
        self.current_best = []

    def calculate_largest(self):
        if self.t >= len(self.current_best):
            self.largest_distance = self.current_best[-1][1]
        else:
            self.largest_distance = self.current_best[self.t-1][1]

    def add(self, point):
        sd = square_distance(point, self.query_point)
        # run through current_best, try to find appropriate place
        for i, e in enumerate(self.current_best):
            if i == self.t:
                return # enough neighbours, this one is farther, let's forget it
            if e[1] > sd:
                self.current_best.insert(i, [point, sd])
                self.calculate_largest()
                return
        # append it to the end otherwise
        self.current_best.append([point, sd])
        self.calculate_largest()
    
    def get_best(self):
        return [element[0] for element in self.current_best[:self.t]]


class MerkleKDTree():
    """ KDTree implementation.
    
        Example usage:
        
            from kdtree import KDTree
            
            data = <load data> # iterable of points (which are also iterable, same length)
            point = <the point of which neighbours we're looking for>
            
            tree = KDTree.construct_from_data(data)
            nearest = tree.query(point, t=4) # find nearest 4 points
    """

    def __init__(self, data):
        dim = 3
        tm_axis = dim
        # print('dim', dim, 'tm_axis', tm_axis)
        def build_kdtree(point_list, depth):
            # code based on wikipedia article: http://en.wikipedia.org/wiki/Kd-tree
            if point_list is None or len(point_list) == 0:
                return None

            # select axis based on depth so that axis cycles through all valid values
            axis = depth % dim # assumes all points have the same dimension

            # sort point list and choose median as pivot point,
            point_list = sorted(point_list, key=lambda point: (point[axis], point[tm_axis]))
            median = len(point_list) // 2 # choose median

            # create node and recursively construct subtrees
            node = KDTreeNode(point=point_list[median],
                              left=build_kdtree(point_list[0:median], depth+1),
                              right=build_kdtree(point_list[median+1:], depth+1))
            return node

        self.dim = dim
        self.root_node = build_kdtree(data, depth=0)

    def compute_merkle_root(self):
        """
        level dependent merkle root
        :return:
        """
        dim = self.dim

        def combine_and_hash(a, b):
            return hasher.sha256((a+b).encode('utf-8')).hexdigest()

        def leveled_hash(node, depth):
            if not node:
                return ''

            l_hash_pair = leveled_hash(node.left, depth+1)
            r_hash_pair = leveled_hash(node.right, depth+1)
            hashed = combine_and_hash(l_hash_pair, r_hash_pair)

            return hashed

        merkle = leveled_hash(self.root_node, 0)
        return merkle
    
    @staticmethod
    def construct_from_data(data):
        tree = MerkleKDTree(data)
        return tree


    def query(self, query_point, count_nn=1):
        # statistics = {'nodes_visited': 0, 'far_search': 0, 'leafs_reached': 0}
        dim = self.dim
        def nn_search(node, query_point, count_nn, depth, best_neighbours):
            if node == None:
                return
            
            #statistics['nodes_visited'] += 1
            
            # if we have reached a leaf, let's add to current best neighbours,
            # (if it's better than the worst one or if there is not enough neighbours)
            if node.is_leaf():
                #statistics['leafs_reached'] += 1
                best_neighbours.add(node.point)
                return
            
            # this node is no leaf
            
            # select dimension for comparison (based on current depth)
            axis = depth % dim
            
            # figure out which subtree to search
            near_subtree = None # near subtree
            far_subtree = None # far subtree (perhaps we'll have to traverse it as well)
            
            # compare query_point and point of current node in selected dimension
            # and figure out which subtree is farther than the other
            if query_point[axis] < node.point[axis]:
                near_subtree = node.left
                far_subtree = node.right
            else:
                near_subtree = node.right
                far_subtree = node.left

            # recursively search through the tree until a leaf is found
            nn_search(near_subtree, query_point, count_nn, depth+1, best_neighbours)

            # while unwinding the recursion, check if the current node
            # is closer to query point than the current best,
            # also, until t points have been found, search radius is infinity
            best_neighbours.add(node.point)
            
            # check whether there could be any points on the other side of the
            # splitting plane that are closer to the query point than the current best
            if (node.point[axis] - query_point[axis])**2 < best_neighbours.largest_distance:
                #statistics['far_search'] += 1
                nn_search(far_subtree, query_point, count_nn, depth+1, best_neighbours)
            
            return
        
        # if there's no tree, there's no neighbors
        if self.root_node != None:
            neighbours = KDTreeNeighbours(query_point, count_nn)
            nn_search(self.root_node, query_point, count_nn, depth=0, best_neighbours=neighbours)
            result = neighbours.get_best()
        else:
            result = []
        
        #print statistics
        return result

    def range(self, min_point, max_point ):
        """Lists the points in the set included in the range defined by the two given points.

        The min_point must have the lower bound for every coordinate, while the max point must have the higher bound.

        :param min_point: the point with coordinates equal to the lowest bounds of the desired range.
        :param max_point: the point with coordinates equal to the highest bounds of the desired range.
        :return: a list with the points of the set falling in the given range.
        """
        return self._range(self.root_node, 0, min_point, max_point)

    def _range(self, node, depth, min_point, max_point):
        if node is None:
            return []

        points = []

        # compare query_point and point of current node in selected dimension
        # and figure out which subtree is farther than the other
        # if query_point[axis] < node.point[axis]:
        #     near_subtree = node.left
        #     far_subtree = node.right
        # else:
        #     near_subtree = node.right
        #     far_subtree = node.left
        dim = self.dim
        def leveled_distance(node, point, depth):
            """Compares two points (the node's point and the given point) on a coordinate based on the node level,
             reporting if the point coordinate is smaller, equal or bigger than the node's point coordinate.

            :param node: the current node
            :param point: the point to add
            :param level: the level of the node
            :return: a negative integer, zero, or a positive integer as the point coordinate is
            less than, equal to, or greater than the node's point coordinate determined by the level.
            """
            axis = depth % dim
            return point[axis] - node.point[axis]

        # Check left side of the tree, if needed
        min_dist = leveled_distance(node, min_point, depth)
        if min_dist <= 0:
            points.extend(self._range(node.left, depth + 1, min_point, max_point))

        # Check right side of the tree, if needed
        max_dist = leveled_distance(node, max_point, depth)
        if max_dist >= 0:
            points.extend(self._range(node.right, depth + 1, min_point, max_point))

        # Return the points found, if the current node's point is NOT inside the range
        for i, coord in enumerate(node.point):
            if not min_point[i] <= coord <= max_point[i]:
                return points

        points.append(node.point)
        return points


def test1(data):
    tree = MerkleKDTree.construct_from_data(data)
    print(tree)

    # find nearest 4 points
    # point = (2, 2)
    # nearest = tree.query(point, t=4)
    # print(nearest)



    # find in range
    min_point = (1, 2, 2)
    max_point = (1,3,4)
    in_range_points = tree.range(min_point, max_point)
    print(in_range_points)
    print()
    print(tree.compute_merkle_root())

def test2(data):
    k = KDTreeNode(data[0], None, None)
    print(k.point_hash)

def test3(data):
    tree = MerkleKDTree.construct_from_data(data)
    print(tree)

    # find nearest 4 points
    # point = (2, 2)
    # nearest = tree.query(point, t=4)
    # print(nearest)

    # k-NN

    q_point = (4, 4, 4)
    knn_points = tree.query(q_point, count_nn=5)

    print(knn_points)

if __name__ == '__main__':
    data = [(1,2,3 ,1234),(2,3,4,1234),(1,0,2,1234),
            (1,4,3,1234), (1,3,5,1234),(4,5,1,1234),
            (2,4,0,1234),(5,1,2,1234),(4,5,4,1234),(4,3,1,1234)]

    test1(data)
