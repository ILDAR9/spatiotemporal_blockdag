import unittest
import Point2D

class Point2DTest(unittest.TestCase):

    def testCreate(self):
        p = Point2D.Point2D(1, 2)
        self.assertEqual(1, p.coords[Point2D.X_COORD])
        self.assertEqual(2, p.coords[Point2D.Y_COORD])

    def testEquality(self):
        p = Point2D.Point2D(1, 2)
        pp = Point2D.Point2D(1, 2)
        p1 = Point2D.Point2D(1, 4)
        p2 = Point2D.Point2D(0, 4)

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

    def testLeveledDist(self):
        p1 = Point2D.Point2D(3, 4)
        p2 = Point2D.Point2D(0, 0)

        self.assertEqual(25, p1.squaredDistanceTo(p2))
        self.assertEqual(5, p1.distanceTo(p2))

if __name__ == '__main__':
    unittest.main()
