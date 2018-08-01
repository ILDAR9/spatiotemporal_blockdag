from rtree import index
import pandas as pd
import numpy as np
import os
import time
from plotly.offline import iplot
from hashlib import sha256
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from itertools import product, combinations, cycle
import pyproj as proj
from utils import get_location, binstr

def read_dataset_eartquakes(fname = 'greek-earthquakes-1964-2000-with-times.csv'):
    data_dir = os.path.join(get_location(), 'data')
    print(data_dir)
    DATASET_FILE = os.path.join(data_dir, fname)
    data = pd.read_csv(DATASET_FILE, delimiter='\t', parse_dates=["DATETIME"])
    # avoid datetime less 1970 year
    data['DATETIME'] = data['DATETIME'].apply(lambda x: x + pd.DateOffset(years=10))
    data['TIMESTAMP'] = data['DATETIME'].values.astype(np.int64) // 10 ** 9
    from datetime import datetime
    print(datetime.utcfromtimestamp(data['TIMESTAMP'].values[0]))
    return data

def read_semantic_dataset():
    data = read_dataset_eartquakes('test-data-with-times.csv')
    def generad_df(x):
        df = data.copy()
        df['LONG'] += 0.1 * x
        df['LAT'] += 0.1 * x
        df['TIMESTAMP'] += x
        return df

    frames = [data] + [generad_df(x) for x in range(-50, 50)]

    result = pd.concat(frames, ignore_index=True)
    return result

class Merkle3DRtree():

    def __init__(self, data=None):
        self.items = {}
        if data is not None:
           self.idx3d = self.__bulk_load(data)
        else:
            self.idx3d = self.__prepaer3Drtree()



    def __bulk_load(self, data, lim = 1600):
        """
        expect columns ('LONG', 'LAT', 'TIMESTAMP') in data
        :param lim: defines number of lines to read from data
        :return: 3D Rtree
        """
        start_time = time.time()

        if lim:
            data = data[:lim]
        # Function required to bulk load
        def generator_function():
            for i, (x, y, tm) in enumerate(data[['LONG', 'LAT', 'TIMESTAMP']].values):
                tm = int(tm)
                i += 100
                # print(100 + i,x,y,tm)
                hash = sha256(binstr(''.join((str(x),str(y),str(tm))))).hexdigest()
                self.items[i] = {'coord':(x,y,tm), 'hash':hash}
                yield (i, (x, x, y, y, tm, tm), hash)

        p = self.__prepare_property()
        idx3d = index.Rtree(generator_function(), properties=p, interleaved=False)
        print('Time elaspsed (s)',time.time() - start_time)

        return idx3d

    def __prepare_property(self):
        # Prepare 3d R-tree and add points
        p = index.Property()
        p.dimension = 3
        p.tight_mbr = True
        # p.buffering_capacity = 100
        # p.dat_extension = 'data'
        # p.idx_extension = 'index'
        # p.leaf_capacity = 128
        # p.region_pool_capacity = 20
        # p.point_pool_capacity = 2
        p.fill_factor = 0.5
        p.variant = index.RT_Linear
        p.index_id = index.RT_RTree
        return p

    def __insert_load(self, data, lim = None):
        """
        expect columns ('LONG', 'LAT', 'TIMESTAMP') in data
        :param lim: defines number of lines to read from data
        :return: 3D Rtree
        """
        start_time = time.time()

        if lim:
            data = data[:lim]
        # Function required to bulk load
        def generator_function():
            for i, (x, y, tm) in enumerate(data[['LONG', 'LAT', 'TIMESTAMP']].values):
                tm = int(tm)
                i += 100
                # print(100 + i,x,y,tm)
                hash = sha256(binstr(''.join((str(x),str(y),str(tm))))).hexdigest()
                yield (i, (x, x, y, y, tm, tm), hash)

        p = self.__prepare_property()
        idx3d = index.Index(properties=p, interleaved=False)
        for x in generator_function():
            idx3d.insert(*x)
        print('Time elaspsed (s)',time.time() - start_time)

        return idx3d

    def list_all_points(self, ):
        # list all the points
        l = [n for n in self.idx3d.intersection((0, 50, 0, 50, 0, 1522894775), objects=True)]
        for n in sorted(l, key=lambda x: x.id):
            print(n.id)
            print('\t', str(n.bbox))
            print('\t', n.object)

    def __prepaer3Drtree(self):
        p = self.__prepare_property()
        idx3d = index.Index(properties=p, interleaved=False)
        return idx3d

    def __convert_to_3d(self, data, R = 6371):
        from numpy import sin, cos, pi
        lat = data['LAT'].values
        lon = data['LONG'].values

        x = sin(pi / 2 - lat) * cos(lon) * R
        y = sin(pi / 2 - lat) * sin(lon) * R
        z = cos(pi / 2 - lat) * R

        return x,y,z

    def plot3d(self, data, lim = 1000, R = 6371):
        '''
        ==============
        3D scatterplot
        ==============
        '''
        df = data[:lim]
        x, y, z = self.__convert_to_3d(df, R)
        trace = go.Scatter3d({"x": x, "y": y, "z": z, 'mode': 'markers'})
        iplot([trace])

    def convert_to_2d(self, data, lim = 1000):
        df = data[:lim]
        lat = df['LAT'].values
        long = df['LONG'].values
        # setup your projections
        crs_wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic
        crs_bng = proj.Proj(init='epsg:27700')  # use a locally appropriate projected CRS
        # then cast your geographic coordinate pair to the projected system
        x, y = proj.transform(crs_wgs, crs_bng, lat, long)
        return x, y

    def get_tree(self):
        leaves = self.idx3d.leaves()
        tree = {}
        bounds = []
        for l in leaves:
            b = tuple(self.idx3d.deinterleave(l[2]))
            bounds.append(b)
            tree[l[0]] = {'items':l[1], 'bounds':b}
        return tree

    def describe(self, lim= None):
        leaves = self.idx3d.leaves()
        if lim: leaves = leaves[:lim]
        bounds = []
        for l in leaves:
            print("%d -> (%d) %s" % (l[0], len(l[1]), l[1]))
            b = tuple(self.idx3d.deinterleave(l[2]))
            print("\t\t%.2f %.2f %.2f %.2f %.0f %.0f" % b)
            bounds.append(b)
        return bounds

    def plot3dTree(self, show_points=False):
        '''
        ==============
        3D scatterplot
        ==============
        '''
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        data = []
        tree = self.get_tree()
        for (idx, mbr), color in zip(tree.items(), cycle(colors)):
            #  BOUNDARY
            bs = tuple(np.reshape(mbr['bounds'], (3, 2)))
            x = []
            y = []
            z = []
            for s, e in combinations(np.array(list(product(*bs))), 2):
                if np.count_nonzero(np.subtract(s, e)) <= 1:
                    x += [s[0], e[0], None]
                    y += [s[1], e[1], None]
                    z += [s[2], e[2], None]
            trace = go.Scatter3d(
                name=str(idx),
                x=x, y=y, z=z,
                marker=dict(
                    size=1
                ),
                line=dict(
                    color=color,
                    width=2,
                )
            )
            data.append(trace)
            # ITEMS
            if show_points:
                coords = np.array([self.items[item]['coord'] for item in mbr['items']]).transpose()

                trace = go.Scatter3d(
                    x=coords[0], y=coords[1], z=coords[2],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=4
                    ),
                    showlegend=False
                )
                data.append(trace)

        layout = go.Layout(title='3D R-tree', width=1024, height=860)
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)

    def merkle(self):
        """
        convert
        """
        pass


    def draw_cube(self, bounds):
        fig = plt.figure( figsize=(16, 8), dpi=200,)
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for bs, color in zip(bounds, cycle(colors)):
            print()
            bs = tuple(np.reshape(bs, (3,2)))
            for s, e in combinations(np.array(list(product(*bs))), 2):
                r = np.count_nonzero(np.subtract(s,e))
                if r <= 1:
                    ax.plot3D(*zip(s, e), color=color)


if __name__ == '__main__':
    data = read_dataset_eartquakes()
    idx3d = Merkle3DRtree()
    idx3d.insert(0, (0, 0, 60))
    idx3d.insert(1, (0, 1, 61))
    res = list(idx3d.intersection((0, 0, 60, 2, 2, 62), objects=True))