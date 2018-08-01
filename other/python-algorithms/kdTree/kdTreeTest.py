import unittest
import kdTree


class KdTreeTest(unittest.TestCase):

    # TEST Point

    def testPointCreate(self):
        p = kdTree.Point([1, 2])
        self.assertEqual(1, p.coords[kdTree.X_COORD])
        self.assertEqual(2, p.coords[kdTree.Y_COORD])

    def testEquality(self):
        p = kdTree.Point([1, 2])
        pp = kdTree.Point([1, 2])
        p1 = kdTree.Point([1, 4])
        p2 = kdTree.Point([0, 4])

        self.assertTrue(p == p)
        self.assertTrue(p == pp)
        self.assertFalse(p == p1)
        self.assertFalse(p == p2)
        self.assertFalse(p1 == p2)

        self.assertFalse(p != p)
        self.assertFalse(p != pp)

        self.assertTrue(p != p1)
        self.assertTrue(p != p2)
        self.assertTrue(p1 != p2)

    def testPointDistance(self):
        p1 = kdTree.Point([3, 4])
        p2 = kdTree.Point([0, 0])

        self.assertEqual(25, p1.squared_distance_to(p2))
        self.assertEqual(5, p1.distance_to(p2))

    # TEST KdTree
    def testIsEmpty(self):
        tree = kdTree.KdTree()
        self.assertTrue(tree.is_empty())

    def testInsert(self):
        tree = kdTree.KdTree()
        self.assertTrue(tree.is_empty())

        p = kdTree.Point([1, 2])
        tree.insert(p)
        self.assertFalse(tree.is_empty())
        self.assertIsNotNone(tree._root)

    def testDimByLevel(self):
        tree = kdTree.KdTree()
        self.assertEqual(2, tree._dim)
        self.assertEqual(0, tree._dim_by_level(0))
        self.assertEqual(1, tree._dim_by_level(1))
        self.assertEqual(0, tree._dim_by_level(2))

        tree = kdTree.KdTree(5)
        self.assertEqual(5, tree._dim)
        self.assertEqual(0, tree._dim_by_level(0))
        self.assertEqual(1, tree._dim_by_level(1))
        self.assertEqual(2, tree._dim_by_level(2))
        self.assertEqual(0, tree._dim_by_level(5))
        self.assertEqual(1, tree._dim_by_level(6))
        self.assertEqual(4, tree._dim_by_level(9))

    def testLeveledDist(self):
        tree = kdTree.KdTree()

        p1 = kdTree.Point([3, 4])
        p2 = kdTree.Point([0, 0])
        node = kdTree.Node(p2)

        self.assertEqual(25, p1.squared_distance_to(p2))
        self.assertEqual(5, p1.distance_to(p2))
        self.assertEqual(3, tree.leveled_distance(node, p1, 0))  # 0 => X compare
        self.assertEqual(4, tree.leveled_distance(node, p1, 1))  # 1 => Y compare
        self.assertEqual(3, tree.leveled_distance(node, p1, 2))  # 0 => X compare

    def testSearch(self):
        p1 = kdTree.Point([3, 4])
        p2 = kdTree.Point([0, 0])
        p3 = kdTree.Point([1, 2])

        tree = kdTree.KdTree()
        tree.insert(p1)
        tree.insert(p2)

        self.assertEqual(p1, tree.search(p1))
        self.assertTrue(tree.contains(p1))

        self.assertEqual(p2, tree.search(p2))
        self.assertTrue(tree.contains(p2))

        self.assertEqual(None, tree.search(p3))
        self.assertFalse(tree.contains(p3))


    def testSearch2(self):
        p1 = kdTree.Point([3, 1, 3])
        p2 = kdTree.Point([4, 4, 2])
        p3 = kdTree.Point([2, 3, 4])

        tree = kdTree.KdTree(dimensions=3)
        tree.insert(p1)
        tree.insert(p2)
        tree.insert(p3)

        self.assertEqual(p1, tree.search(p1))
        self.assertEqual(p2, tree.search(p2))
        self.assertEqual(p3, tree.search(p3))

        p = kdTree.Point([1, 2, 3])
        self.assertEqual(None, tree.search(p))

        self.assertTrue(tree.contains(p1))
        self.assertTrue(tree.contains(p2))
        self.assertTrue(tree.contains(p3))
        self.assertFalse(tree.contains(p))


    def testSearch3(self):
        p = kdTree.Point([1, 2])
        pp = kdTree.Point([1, 2])

        tree = kdTree.KdTree()
        tree.insert(p)

        self.assertEqual(p, tree.search(pp))
        self.assertTrue(tree.search(pp) is p, "Returned point should be the node's point")
        self.assertFalse(tree.search(pp) is pp, "Returned point should be the node's point, not the given point")

    def testNearest(self):
        p1 = kdTree.Point([3, 1])
        p2 = kdTree.Point([4, 4])
        p3 = kdTree.Point([2, 3])
        p4 = kdTree.Point([0.5, 0.5])

        tree = kdTree.KdTree()
        tree.insert(p1)
        tree.insert(p2)
        tree.insert(p3)
        tree.insert(p4)
        self.assertEqual(4, tree.size())

        p = kdTree.Point([1, 2])
        nn = tree.nearest(p)
        self.assertIsNotNone(nn)
        self.assertEqual(p3, nn)

        p = kdTree.Point([5, 5])
        nn = tree.nearest(p)
        self.assertIsNotNone(nn)
        self.assertEqual(p2, nn)

    def testNearest3D(self):
        p1 = kdTree.Point([3, 1, 0])
        p2 = kdTree.Point([4, 4, 0])
        p3 = kdTree.Point([2, 3, 1])
        p4 = kdTree.Point([0.5, 0.5, 10])

        tree = kdTree.KdTree(dimensions=3)
        tree.insert(p1)
        tree.insert(p2)
        tree.insert(p3)
        tree.insert(p4)
        self.assertEqual(4, tree.size())

        p = kdTree.Point([2, 2, 1])
        nn = tree.nearest(p)
        self.assertIsNotNone(nn)
        self.assertEqual(p3, nn)

        p = kdTree.Point([2, 2, 8])
        nn = tree.nearest(p)
        self.assertIsNotNone(nn)
        self.assertEqual(p4, nn)

    def testRange(self):
        p1 = kdTree.Point([3, 1])
        p2 = kdTree.Point([4, 4])
        p3 = kdTree.Point([2, 3])
        p4 = kdTree.Point([0.5, 0.5])

        tree = kdTree.KdTree()
        tree.insert(p1)
        tree.insert(p2)
        tree.insert(p3)
        tree.insert(p4)

        points = tree.range(kdTree.Point([0, 0]), kdTree.Point([1, 1]))
        self.assertEqual(1, len(points))
        self.assertTrue(p4 in points)

        points = tree.range(kdTree.Point([1, 2]), kdTree.Point([5, 5]))
        self.assertEqual(2, len(points))
        self.assertTrue(p2 in points)
        self.assertTrue(p3 in points)

if __name__ == '__main__':
    unittest.main()
