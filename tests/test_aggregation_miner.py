import unittest

from moveminerx.mining import PointClusterMiner, LineClusterMiner
from moveminerx.mining._basic_class import TrajectoryPoint


class TestAggregationMiner(unittest.TestCase):
    def test_point_cluster_dbscan(self):
        points = [
            TrajectoryPoint(0.0, 0.0, time=0, oid='a'),
            TrajectoryPoint(0.5, 0.1, time=0, oid='b'),
            TrajectoryPoint(10.0, 10.0, time=0, oid='c')
        ]
        miner = PointClusterMiner(eps=1.0, min_samples=2)
        miner.fit(points, method='DBSCAN')
        patterns = miner.get_patterns()

        self.assertEqual(len(patterns), 1)
        self.assertEqual({p.oid for p in patterns[0]}, {'a', 'b'})

    def test_point_cluster_kmeans(self):
        points = [
            TrajectoryPoint(0.0, 0.0, time=0, oid='a'),
            TrajectoryPoint(0.5, 0.1, time=0, oid='b'),
            TrajectoryPoint(10.0, 10.0, time=0, oid='c'),
            TrajectoryPoint(10.1, 9.9, time=0, oid='d')
        ]
        miner = PointClusterMiner()
        miner.fit(points, method='KMeans', n_clusters=2)
        patterns = miner.get_patterns()

        self.assertEqual(len(patterns), 2)
        all_points = {p.oid for cluster in patterns for p in cluster}
        self.assertEqual(all_points, {'a', 'b', 'c', 'd'})

    def test_line_cluster_dbscan(self):
        lines = [
            ((0.0, 0.0), (1.0, 1.0)),
            ((0.0, 1.0), (1.0, 2.0)),
            ((10.0, 10.0), (11.0, 11.0))
        ]
        miner = LineClusterMiner()
        miner.fit(lines, method='DBSCAN', eps=2.0, min_samples=2, metric='euclidean')
        patterns = miner.get_patterns()

        self.assertEqual(len(patterns), 1)
        self.assertEqual(len(patterns[0]), 2)

    def test_line_cluster_precomputed(self):
        lines = [
            ((0.0, 0.0), (1.0, 1.0)),
            ((0.0, 1.0), (1.0, 2.0)),
            ((10.0, 10.0), (11.0, 11.0))
        ]
        miner = LineClusterMiner()
        miner.fit(lines, method='DBSCAN', eps=2.0, min_samples=2, metric='precomputed')
        patterns = miner.get_patterns()

        self.assertEqual(len(patterns), 1)
        self.assertEqual(len(patterns[0]), 2)


if __name__ == '__main__':
    unittest.main()
