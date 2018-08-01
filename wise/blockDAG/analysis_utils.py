#!/usr/bin/python
# encoding: utf-8

import time
import pandas as pd
from blockDAG import search_on_blockDAG as sob
from .geo_utils import to_Cartesian, kmToDIST
from simulation import GeneratorDAGchain
from utils import get_ts
import os, errno

def __measure_time(search_f, range_count, *args):
    postfix = ""
    if type(args[-1]) == bool:
        postfix = "(tm)" if args[-1] else "(sp)"

    rows = []

    col_indx = 'indx'
    col_res_count = search_f.__name__ + postfix + '_res_count'
    col_comp_t = search_f.__name__ + postfix + '_comp_t'
    for i in range(range_count):
        ################
        st = time.time()
        res_count = len(search_f(*args))
        comp_t = time.time() - st
        ################
        row = {
            col_indx : i,
            col_res_count : res_count,
            col_comp_t : comp_t,
        }
        rows.append(row)

    df = pd.DataFrame.from_dict(rows)[[col_indx, col_res_count, col_comp_t]]
    return df


def __print_advantage(df, pref, pref_scan, pref_scan2 = None):

    key1 = pref + '_comp_t'
    key2 = pref_scan + '_comp_t'
    df['dt_diff_scan(tm)'] = df[key2] - df[key1]
    df['dt_relative_scan(tm)'] = df[key2] / df[key1]

    print("Scan (time)", df[key1].mean(), df[key2].mean(),
          df[key2].mean() - df[key1].mean())

    if pref_scan2:
        key2 = pref_scan2 + '_comp_t'
        df['dt_diff_scan(sp)'] = df[key2] - df[key1]
        df['dt_relative_scan(sp)'] = df[key2] / df[key1]
        print("Scan (space)", df[key1].mean(), df[key2].mean(),
              df[key2].mean() - df[key1].mean())

def __to_Cartesian_rect(p1, p2):
    p1 = to_Cartesian(*p1)
    p2 = to_Cartesian(*p2)
    return (min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2])), (max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2]))


######################################
# range, ps: scan time and space first
######################################
def compare_time_range(block_dag, min_point, max_point, t_start, t_end, range_count=5, debug=True):
    assert min_point[0] <= max_point[0] and min_point[1] <= max_point[1] and t_start < t_end
    min_point, max_point = __to_Cartesian_rect(min_point, max_point)
    rng = __measure_time(sob.kd_range, range_count, block_dag, min_point, max_point, t_start, t_end)
    rng_scan = __measure_time(sob.scan_range, range_count, block_dag, min_point, max_point, t_start, t_end, True)
    rng_scan2 = __measure_time(sob.scan_range, range_count, block_dag, min_point, max_point, t_start, t_end, False)

    df = pd.merge(rng, rng_scan, on='indx').merge(rng_scan2, on='indx')
    if debug:
        __print_advantage(df, pref = sob.kd_range.__name__, pref_scan = sob.scan_range.__name__ + "(tm)", pref_scan2 = sob.scan_range.__name__ + "(sp)")

    return df


######
# k-NN
######
def compare_time_knn(block_dag, q_point, count_nn, t_start, t_end, range_count=5, debug=True):
    assert t_start < t_end
    q_point = to_Cartesian(*q_point)
    rng = __measure_time(sob.kd_knn, range_count, block_dag, q_point, count_nn, t_start, t_end)
    rng_scan = __measure_time(sob.scan_knn, range_count, block_dag, q_point, count_nn, t_start, t_end)

    df = pd.merge(rng, rng_scan, on='indx')
    if debug:
        __print_advantage(df, pref = sob.kd_knn.__name__, pref_scan = sob.scan_knn.__name__)

    return df


############
# k-NN bound
############
def compare_time_knn_bound(block_dag, q_point, count_nn, bound, t_start, t_end, range_count=5, debug = True):
    assert t_start < t_end and bound > 0 and count_nn > 0
    q_point = to_Cartesian(*q_point)
    bound = kmToDIST(bound)
    rng = __measure_time(sob.kd_knn_bound, range_count, block_dag, q_point, count_nn, bound, t_start, t_end)
    rng_scan = __measure_time(sob.scan_knn_bound, range_count, block_dag, q_point, count_nn, bound, t_start, t_end)

    df = pd.merge(rng, rng_scan, on='indx')
    if debug:
        __print_advantage(df, pref = sob.kd_knn_bound.__name__, pref_scan = sob.scan_knn_bound.__name__)

    return df


###########################################
# Ball query, ps: scan time and space first
###########################################
def compare_time_query_ball(block_dag, q_point, r, t_start, t_end, range_count=5, debug = True):
    assert t_start < t_end and r > 0
    q_point = to_Cartesian(*q_point)
    r = kmToDIST(r)
    rng = __measure_time(sob.kd_query_ball, range_count, block_dag, q_point, r, t_start, t_end)
    rng_scan = __measure_time(sob.scan_query_ball, range_count, block_dag, q_point, r, t_start, t_end, True)
    rng_scan2 = __measure_time(sob.scan_query_ball, range_count, block_dag, q_point, r, t_start, t_end, False)

    df = pd.merge(rng, rng_scan, on='indx').merge(rng_scan2, on='indx')
    if debug:
        __print_advantage(df, pref = sob.kd_query_ball.__name__, pref_scan = sob.scan_query_ball.__name__ + "(tm)", pref_scan2 = sob.scan_query_ball.__name__ + "(sp)")

    return df

###############################################################
def save_df(df, name, index = False):
    dir_fname = os.path.join('../plot_scripts/data_plot/', name)
    print(dir_fname)
    if not os.path.exists(dir_fname):
        try:
            os.makedirs(os.path.dirname(dir_fname))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    df.to_csv(dir_fname, index = index)

cols_remove_dict = {'range':  ['indx', 'kd_range_res_count', 'scan_range(tm)_res_count', 'scan_range(sp)_res_count'],
                   'knn' : ['indx',	'kd_knn_res_count', 'scan_knn_res_count'],
                   'knn_bound' : ['indx', 'kd_knn_bound_res_count', 'scan_knn_bound_res_count'],
                   'query_ball' : ['indx', 'kd_query_ball_res_count', 'scan_query_ball(tm)_res_count', 'scan_query_ball(sp)_res_count']
                   }

###############################
# Growing block-DAG experiments
###############################
def experiment_range_growing_blockchain(range_count=4, algo = 'range'):
    def settings_size_generator(max_mult):
        for mult in range(50, max_mult + 1, 50):
            yield dict(repeat_times=mult, tr=70, D=3, bs=50, alpha=20)

    cols_remove = cols_remove_dict[algo]
    # mn = (22.20, 114.03)
    # mx = (22.31, 114.05)
    mn = (22.20, 113.89)
    mx = (22.31, 113.95)
    q_point = (22.6, 114)
    t_start = get_ts(2018, 1, 1, 0, 0, 0)
    t_end = get_ts(2018, 1, 1, 0, 0, 0)

    kwargs = dict(range_count=range_count, debug=False)
    def execute(block_dag):

        t_start = block_dag.chain[1]['timestamp']
                # + (block_dag.chain[-1]['timestamp'] - block_dag.chain[1]['timestamp']) // 2
        hours = 2
        t_end = t_start + int(60 * 60 * hours)

        print(algo, (t_end - t_start) / (60 * 60), 'hours')
        df = None
        if algo == 'range':
            block_dag.optimize()
            df = compare_time_range(block_dag, mn, mx, t_start, t_end, **kwargs)
        elif algo == 'knn':
            block_dag.super_optimize()
            count_nn = 15
            df = compare_time_knn(block_dag, q_point, count_nn, t_start, t_end, **kwargs)
        elif algo == 'knn_bound':
            block_dag.super_optimize()
            bound = 12
            count_nn = 15
            df = compare_time_knn_bound(block_dag, q_point, count_nn, bound, t_start, t_end, **kwargs)
        elif algo == 'query_ball':
            block_dag.super_optimize()
            r = 20
            df = compare_time_query_ball(block_dag, q_point, r, t_start, t_end, **kwargs)
        df.drop(cols_remove, axis=1, inplace=True)
        df = df * 1000
        return df

    dfs = []
    for settings in settings_size_generator(301):
        print("=============={}==============".format(settings['bs']))
        block_dag = GeneratorDAGchain.generate(**settings)
        df = execute(block_dag)
        df['trx_count'] = 18732 * settings['repeat_times']
        dfs.append(df)
        del block_dag
    df = pd.concat(dfs).groupby('trx_count').median()
    df.plot(title='growing block-DAG')
    save_df(df, '{}/{}_growing.csv'.format(algo, algo), True)
    return df

###########################################
# Spatiotemporal and block-size experiments
###########################################
def experiment_bs_and_spatiotemporal(range_count, algo = 'range'):

    def settings_generator(bs_mn, bs_max, bs_step = 1):
        for bs in range(bs_mn, bs_max + 1, bs_step):
            yield dict(repeat_times=300, tr=bs + 30, D=3, bs=bs, alpha=20)
    dfs_bs = []
    dfs = []

    cols_remove = cols_remove_dict[algo]

    # mn = (22.20, 114.03)
    # mx = (22.31, 114.05)
    mn = (22.20, 113.89)
    mx = (22.31, 113.95)
    q_point = (22.6, 114)

    kwargs = dict(range_count=range_count, debug=False)
    def execute(block_dag, t_start, t_end):
        print(algo, (t_end - t_start) / (60 * 60), 'hours')
        df = None
        if algo == 'range':
            block_dag.optimize()
            df = compare_time_range(block_dag, mn, mx, t_start, t_end, **kwargs)
        elif algo == 'knn':
            block_dag.super_optimize()
            count_nn = 15
            df = compare_time_knn(block_dag, q_point, count_nn, t_start, t_end, **kwargs)
        elif algo == 'knn_bound':
            block_dag.super_optimize()
            bound = 12
            count_nn = 15
            df = compare_time_knn_bound(block_dag, q_point, count_nn, bound, t_start, t_end, **kwargs)
        elif algo == 'query_ball':
            block_dag.super_optimize()
            r = 20
            df = compare_time_query_ball(block_dag, q_point, r, t_start, t_end, **kwargs)
        df.drop(cols_remove, axis=1, inplace=True)
        df = df * 1000
        return df

    bs_consider = [40, 100]

    for settings in settings_generator(30, 110, bs_step=10):
        print("=============={}==============".format(settings['bs']))
        block_dag = GeneratorDAGchain.generate(**settings)
        # block size experiment
        t_start = get_ts(2017, 1, 2, 0, 0, 0)
        t_end = get_ts(2019, 1, 3, 0, 0, 0)

        df_res = execute(block_dag, t_start, t_end)
        df_res['bs'] = settings['bs']
        dfs_bs.append(df_res)

        if settings['bs'] not in bs_consider:
            continue

        # spatiotemporal query
        t_start = block_dag.chain[1]['timestamp']
                  # + (block_dag.chain[-1]['timestamp'] - block_dag.chain[1]['timestamp']) // 2
        for hours in [0.5] + list(range(1, 12, 1)):
            t_end = t_start + int(60 * 60 * hours)

            df = execute(block_dag, t_start, t_end)
            df['bs'] = settings['bs']
            df['hours'] = hours
            dfs.append(df)
        del block_dag

    # trnasform to milliseconds
    df_bs = pd.concat(dfs_bs).groupby('bs').median()
    df = pd.concat(dfs).groupby(['bs', 'hours'], as_index=False).median()

    df_bs.plot(title='per bs')
    for bs in bs_consider:
        df[df['bs'] == bs].drop(['bs'], axis=1).plot(x='hours', title='bs: ' + str(bs))

    save_df(df_bs, '{}/{}_bs.csv'.format(algo, algo), True)
    save_df(df, '{}/{}_spatiotemporal.csv'.format(algo, algo))
    return df_bs, df

#########################
# k-NN vary k experiments
#########################
def experiment_knn_vary_k(range_count):
    algo = 'knn'
    dfs = []

    q_point = (22.6, 114)
    cols_remove = cols_remove_dict[algo]
    kwargs = dict(range_count=range_count, debug=False)

    bs = 50
    settings = dict(repeat_times=300, tr=70, D=3, bs=bs, alpha=20)
    block_dag = GeneratorDAGchain.generate(**settings)
    block_dag.super_optimize()
    for count_nn in range(2, bs+3,2):
        print("=============={}==============".format(count_nn))

        # block size experiment
        t_start = get_ts(2017, 1, 2, 0, 0, 0)
        t_end = get_ts(2019, 1, 3, 0, 0, 0)

        df = compare_time_knn(block_dag, q_point, count_nn, t_start, t_end, **kwargs)
        df.drop(cols_remove, axis=1, inplace=True)
        df = df * 1000

        df['k'] = count_nn
        dfs.append(df)

    df_k = pd.concat(dfs).groupby('k').median()

    df_k.plot(title='per k')

    save_df(df_k, '{}/{}_vary_k.csv'.format(algo, algo), True)
    return df_k


#############################
# k-NN vary bound experiments
#############################
def experiment_knn_bound_vary_b(range_count):
    algo = 'knn_bound'
    dfs = []

    q_point = (22.6, 114)
    cols_remove = cols_remove_dict[algo]
    kwargs = dict(range_count=range_count, debug=False)

    bs = 50
    settings = dict(repeat_times=300, tr=70, D=3, bs=bs, alpha=10)
    block_dag = GeneratorDAGchain.generate(**settings)
    block_dag.super_optimize()
    count_nn = 25
    for bound in range(1, 12, 1):
        print("=============={}==============".format(bound))

        # block size experiment
        t_start = get_ts(2017, 1, 2, 0, 0, 0)
        t_end = get_ts(2019, 1, 3, 0, 0, 0)

        df = compare_time_knn_bound(block_dag, q_point, count_nn, bound, t_start, t_end, **kwargs)
        df.drop(cols_remove, axis=1, inplace=True)
        df = df * 1000

        df['bound'] = bound
        dfs.append(df)

    df_k = pd.concat(dfs).groupby('bound').median()

    df_k.plot(title='per bound')

    save_df(df_k, '{}/{}_vary_bound.csv'.format(algo, algo), True)
    return df_k


###############################
# ball-point vary r experiments
###############################
def experiment_ball_point_vary_r(range_count):
    algo = 'query_ball'
    dfs = []

    q_point = (22.6, 114)
    cols_remove = cols_remove_dict[algo]
    kwargs = dict(range_count=range_count, debug=False)

    bs = 50
    settings = dict(repeat_times=300, tr=75, D=3, bs=bs, alpha=10)
    block_dag = GeneratorDAGchain.generate(**settings)
    block_dag.super_optimize()
    for r in range(1, 50, 3):
        print("=============={}==============".format(r))

        # block size experiment
        t_start = get_ts(2017, 1, 2, 0, 0, 0)
        t_end = get_ts(2019, 1, 3, 0, 0, 0)

        df = compare_time_query_ball(block_dag, q_point, r, t_start, t_end, **kwargs)
        df.drop(cols_remove, axis=1, inplace=True)
        df = df * 1000

        df['r'] = r
        dfs.append(df)

    df_k = pd.concat(dfs).groupby('r').median()

    df_k.plot(title='per r')

    save_df(df_k, '{}/{}_vary_r.csv'.format(algo, algo), True)
    return df_k


