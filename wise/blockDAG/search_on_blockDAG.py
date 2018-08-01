#!/usr/bin/python
# encoding: utf-8
from merkleKDtree.merkle_kdtree import square_distance
from itertools import chain, islice
from heapq import heappush, heappop
from .geo_utils import to_Cartesian
from heapq import nsmallest


####################
# only for blockDAG (self)
####################
def kdtree_indxes_gen(self, t_start, t_end):
    """
    BFS filter by time and k-NN query for each kd tree found
    """
    orphan_hashes = self.orphan_hashes
    stop_search = False
    # Temporal search
    while not stop_search and orphan_hashes:
        max_end_time = 0
        for hs in orphan_hashes:
            bh = self.hash_map_block[hs]
            # predicate
            end_t = bh['end_time']
            if end_t >= t_start and end_t <= t_end:
                yield bh['index']

            max_end_time = max(max_end_time, end_t)
        # No reason to continue BFS
        stop_search = max_end_time < t_start
        orphan_hashes = bh['orphan_hashes']


#-------------
# Range search
#-------------
def kd_range(self, min_point, max_point, t_start, t_end=None):
    results = []
    for indx in kdtree_indxes_gen(self, t_start, t_end):
        results += self.merkle_kd_trees[indx].range(min_point, max_point)

    return results


#-----
# K-NN
#-----
def kd_knn(self, q_point, count_nn, t_start, t_end):
    results = []

    for indx in kdtree_indxes_gen(self, t_start, t_end):
        # if count_nn >=  tree.data.shape[0]:
        #     # for item in map(lambda coord: (float(square_distance(q_point, coord)), tuple(coord)), tree.data):
        #         # heappush(results, item)
        #     # continue
        #     results += list(map(lambda coord: (float(square_distance(q_point, coord)), tuple(coord)), tree.data))
        # else:

        candidates = self.merkle_kd_trees[indx].query(q_point, count_nn)
        results += list(zip(*candidates)) if type(candidates[1]) != int else [candidates]
        # if type(candidates[1]) == int:
        #     heappush(results, candidates)
        # else:
        #     for item in zip(*candidates):
        #         heappush(results, item)

    # return nsmallest(count_nn, results, key=lambda tr: tr[0])
    return [heappop(results) for i in range(count_nn)]


#-------------
# K-NN limited
#-------------
def kd_knn_bound(self, q_point, count_nn, distance_upper_bound, t_start, t_end):
    results = []

    for indx in kdtree_indxes_gen(self, t_start, t_end):
        bs = self.chain[indx]['trx_count']
        candidates = self.merkle_kd_trees[indx].query(q_point, count_nn, distance_upper_bound = distance_upper_bound)
        if type(candidates[1]) == int:
            heappush(results, candidates)
        else:
            for item in zip(*candidates):
                if item[1] != bs:
                    heappush(results, item)
                    # results.append(item)

    # return sorted(results, key = lambda item: item[0])[:count_nn]
    return [heappop(results) for i in range(count_nn)] if len(results) > count_nn else results


#-----------
# Ball-point
#-----------
def kd_query_ball(self, q_point, r, t_start, t_end):
    results = []

    for indx in kdtree_indxes_gen(self, t_start, t_end):
        results += self.merkle_kd_trees[indx].query_ball_point(q_point, r)

    return results


##############
# SCAN based #
##############
def block_headers_gen(self):

    orphan_hashes = self.orphan_hashes

    while orphan_hashes:
        for hs in orphan_hashes:
            bh = self.hash_map_block[hs]
            yield bh

        orphan_hashes = bh['orphan_hashes']


#------------
# scan: range
#------------
def scan_range(self, mn, mx, t_start, t_end, is_time_first = True):
    """
    Scan every transactions for spatiotemporal search
    """
    results = []
    f_in_t = lambda x: t_start <= x and x <= t_end
    f_in_s = lambda x: mn[0] <= x[0] and mn[1] <= x[1] and mn[2] <= x[2] \
                       and x[0] <= mx[0] and x[1] <= mx[1] and x[2] <= mx[2]

    if is_time_first:
        for bh in block_headers_gen(self):
            results += list(filter(lambda tr: f_in_t(tr['ts']), bh['transactions']))
        results = list(filter(lambda tr: f_in_s(to_Cartesian(tr['lat'], tr['lon'])), results))
    else:
        for bh in block_headers_gen(self):
            results += list(filter(lambda tr: f_in_s(to_Cartesian(tr['lat'], tr['lon'])), bh['transactions']))
        results = list(filter(lambda tr: f_in_t(tr['ts']), results))

    return results

#----------
# scan: knn
#----------
def scan_knn(self, q_point, count_nn, t_start, t_end=None):
    """
    Scan every transactions for spatiotemporal search
    """
    results = []

    f_in_t = lambda x: t_start <= x and x <= t_end
    for bh in block_headers_gen(self):
        candidates = filter(lambda tr: f_in_t(tr['ts']), bh['transactions'])
        candidates = map(lambda tr: (to_Cartesian(tr['lat'], tr['lon']), tr), candidates)
        results += list(candidates)

    results = nsmallest(count_nn, results, key=lambda tr: square_distance(q_point, tr[0]))
    return results

#-------------------
# scan: K-NN bound
#-------------------
def scan_knn_bound(self, q_point, count_nn, bound, t_start, t_end):
    """
    Scan every transactions for spatiotemporal search
    """
    results = []
    f_in_t = lambda x: t_start <= x and x <= t_end

    for bh in block_headers_gen(self):
        candidates = filter(lambda tr: f_in_t(tr['ts']), bh['transactions'])
        # candidates = bh['transactions']
        candidates = map(lambda tr: (to_Cartesian(tr['lat'], tr['lon']), tr), candidates)
        results += list(candidates)

    results = filter(lambda tr: square_distance(q_point, tr[0]  ) <= (bound**2), results)
    results = nsmallest(count_nn, results, key=lambda tr: square_distance(q_point, tr[0]))
    # results = sorted(results, key=lambda tr: square_distance(q_point, tr[0]))[:count_nn]

    return results

#-----------------
# scan: Ball-point
#-----------------
def scan_query_ball(self, q_point, r, t_start, t_end, is_time_first = True):
    results = []
    f_in_t = lambda tr: t_start <= tr['ts'] and tr['ts'] <= t_end
    f_in_s = lambda tr: square_distance(q_point, to_Cartesian(tr['lat'], tr['lon'])) <= r**2
    if is_time_first:
        for bh in block_headers_gen(self):
            candidates = filter(f_in_t, bh['transactions'])
            results += list(candidates)
        results = list(filter(f_in_s, results))
    else:
        for bh in block_headers_gen(self):
            candidates = filter(f_in_s, bh['transactions'])
            results += list(candidates)

        results += list(filter(f_in_t, results))

    return results