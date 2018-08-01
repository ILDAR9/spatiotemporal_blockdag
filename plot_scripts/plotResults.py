#!/usr/bin/env python
import pylab as pl
import os
import pandas as pd

FLG_SAVE_IMG = True
FLG_SHOW = False

DATA_FOLDER_NAME = ""
def update_data_folder(query_name ="range"):
    global DATA_FOLDER_NAME
    DATA_FOLDER_NAME = os.path.join('data_plot', query_name)

OUT_FOLDER_NAME = 'out'
MAX_X = 20000
MAX_Y = 25000


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for row in f:
            a = tuple(float(num) for num in row.split())
            if a[0] <= MAX_X: data.append(a)
    return data


def set_up(query_name):
    def set_up_folders():
        check_and_create_foldeer = lambda dname: os.makedirs(dname) if not os.path.exists(dname) else None
        check_and_create_foldeer(OUT_FOLDER_NAME)
        check_and_create_foldeer(os.path.join(OUT_FOLDER_NAME, query_name))

    if FLG_SAVE_IMG: set_up_folders()


def plot_data(query_name):
    update_data_folder(query_name)

    pref_k1 = 'mkd_'
    pref_k2 = 'scantime_'
    pref_k3 = 'scanspace_'
    pref_k4 = '?_'
    d = {pref_k1 + 'label': 'Merkle KD-tree',
         pref_k1 + 'color': 'tab:green',
         pref_k1 + 'marker': 'o',
         pref_k2 + 'label': 'SCAN time-space',
         pref_k2 + 'color': 'tab:blue',
         pref_k2 + 'marker': 'x',
         pref_k3 + 'label': 'SCAN space-time',
         pref_k3 + 'color': 'tab:orange',
         pref_k3 + 'marker': 'D',
         pref_k4 + 'label': '',
         pref_k4 + 'color': 'red',
         pref_k4 + 'marker': 's',
         'xlabel': '',
         'axes_size' : 22
         }

    def set_up_plot():
        TITLE_SIZE = 22  # 14
        LEGEND_SIZE = 18 # 14
        TIC_SIZE = 16    # 12
        # TIC_LABEL_SIZE = 16

        pl.xticks(fontsize=TIC_SIZE)
        pl.yticks(fontsize=TIC_SIZE)
        params = {'legend.fontsize': LEGEND_SIZE, # Line names
                  # 'figure.figsize': (20, 10),
                  # 'axes.labelsize': TIC_LABEL_SIZE,
                  'axes.titlesize': TITLE_SIZE, # Plot title
                  # 'xtick.labelsize': TIC_LABEL_SIZE,
                  # 'ytick.labelsize': TIC_LABEL_SIZE
                  }
        pl.rcParams.update(params)

        axes = pl.axes()
        axes.autoscale_view()
        axes.set_title(d['title'], )
        axes.title.set_position([.5, 1.05])
        pl.legend(loc='upper right', fancybox=True, shadow=True)
        pl.xlim(xmin=-10)
        pl.ylim(ymin=-0.03)

    #--------
    # Figures
    #--------
    def plot_of(rng):
        axis_font = {'fontname': 'Arial', 'size': d['axes_size']}
        pl.xlabel(d['xlabel'], **axis_font)
        pl.ylabel(d['ylabel'], **axis_font)
        xmax = 0
        for prefix in (pref_k1, pref_k2, pref_k3)[:rng]:
            # vals = tuple(zip(d['xdata'], d[prefix + 'data']))
            xmax = d['xdata'].max()
            pl.plot(d['xdata'], d[prefix + 'data'], markersize=10, marker=d[prefix + 'marker'], label=d[prefix + 'label'],
                    color=d[prefix + 'color'], linewidth=3)

        set_up_plot()
        pl.grid()
        pl.xlim(xmax=(xmax + d['xmargin']))
        pl.xlim(xmin=d['xmin'])
        pl.ylim(ymin=d['ymin'])
        pl.gcf().canvas.set_window_title(d['ylabel'])
        pl.axes().legend(loc=d['loc'], shadow=True, ncol=1)
        pl.tight_layout()
        if FLG_SAVE_IMG: pl.savefig(os.path.join(OUT_FOLDER_NAME, query_name, d['img_name']))
        if FLG_SHOW: pl.show()
        pl.gcf().clear()

    query_name_full = {'range': 'Range  Query', 'knn' : 'k-NN',
                      'knn_bound' : 'Bounded k-NN', 'query_ball' : 'Ball Query'}[query_name]

    #############
    # Block size
    #############
    def plot_block_size(query_name):

        df = pd.read_csv(os.path.join(DATA_FOLDER_NAME, query_name + '_bs.csv'))

        d.update({
            pref_k1 + 'data': df['kd_{}_comp_t'.format(query_name)].values,
            pref_k2 + 'data': df['scan_{}{}_comp_t'.format(query_name,
                                            ('(tm)' if 'knn' not in query_name else ''))].values,
            pref_k3 + 'data': df['scan_{}(sp)_comp_t'.format(query_name)].values if 'knn' not in query_name else None,
            'xdata': df['bs'].values,
            'ylabel': 'Time (ms)',
            'xlabel': 'Block size',
            'img_name': 'blocksize_{}.pdf'.format(query_name),
            'title': '{} ( 2 weeks )'.format(query_name_full),
            'ymax': 230,
            'xmargin': 2,
            'ymin': 0,
            'xmin': 28,
            'loc': 'center right'})
        plot_of(3 if 'knn' not in query_name else 2)

    ################
    # Spatiotemporal
    ################
    def plot_spatiotemporal(query_name, bs=40):

        df = pd.read_csv(os.path.join(DATA_FOLDER_NAME, query_name + '_spatiotemporal.csv'))
        df = df[df['bs'] == bs]

        loc = None
        if 'knn' not in query_name:
            if bs == 40:
                loc = (0.049, 0.5)
            elif 'range' in query_name:
                loc = (0.43, 0.2)
            else:
                loc = (0.4, 0.12)
        else:
            loc = (0.008, 0.8)

        d.update({
            pref_k1 + 'data': df['kd_{}_comp_t'.format(query_name)].values,
            pref_k2 + 'data': df['scan_{}{}_comp_t'.format(query_name, ('(tm)' if 'knn' not in query_name else '')
                                                           )].values,
            pref_k3 + 'data': df['scan_{}(sp)_comp_t'.format(query_name)].values if 'knn' not in query_name else None,
            'xdata': df['hours'].values,
            'ylabel': 'Time (ms)',
            'xlabel': 'Temporal search range ( hours )',
            'img_name': 'spatiotemporal_bs_{}_{}.pdf'.format(bs, query_name),
            'title': '{} ( bs={} )'.format(query_name_full, bs),
            'ymax': 230,
            'xmargin': 1,
            'ymin': -300,
            'xmin': 0,
            'loc': loc
        })

        plot_of(3 if 'knn' not in query_name else 2)

    #########
    # Growing
    #########
    def plot_growing(query_name):
        df = pd.read_csv(os.path.join(DATA_FOLDER_NAME, query_name + '_growing.csv'))
        d.update({
            pref_k1 + 'data': df['kd_{}_comp_t'.format(query_name)].values,
            pref_k2 + 'data': df['scan_{}{}_comp_t'.format(query_name, ('(tm)' if 'knn' not in query_name else '')
                                                           )].values,
            pref_k3 + 'data': df['scan_{}(sp)_comp_t'.format(query_name)].values if 'knn' not in query_name else None,
            'xdata': df['trx_count'].values // 1000,
            'ylabel': 'Time (ms)',
            'xlabel': 'Transactions stored (x1000)',
            'img_name': 'growing_{}.pdf'.format(query_name),
            'title': '{} on growing Blockchain'.format(query_name_full),
            'ymax': 290,
            'xmargin': 200, #5500
            'ymin': -8,
            'xmin': 180, #12000
            'loc': (0.01, 0.69) if 'knn' not in query_name else (0.01, 0.77) #"upper left"
        })
        plot_of(3 if 'knn' not in query_name else 2)

    ############
    # KNN vary k
    ############
    def plot_knn_vary_k(query_name):
        df = pd.read_csv(os.path.join(DATA_FOLDER_NAME, query_name + '_vary_k.csv'))
        df = df.iloc[::3, :]
        d.update({
            pref_k1 + 'data': df['kd_{}_comp_t'.format(query_name)].values,
            pref_k2 + 'data': df['scan_{}_comp_t'.format(query_name)].values,
            'xdata': df['k'].values,
            'ylabel': 'Time (ms)',
            'xlabel': 'K',
            'img_name': 'knn_vary_k.pdf',
            'title': '{} ( 2 weeks )'.format(query_name_full),
            'ymax': 480,
            'xmargin': 2,
            'ymin': -5,
            'xmin': -1,
            'loc': (0.33, 0.5)
        })
        plot_of(2)

    ######################
    # KNN bound vary bound
    ######################
    def plot_knn_bound_vary_bound(query_name):
        df = pd.read_csv(os.path.join(DATA_FOLDER_NAME, query_name + '_vary_bound.csv'))
        # df = df[:50].iloc[::3, :]
        d.update({
            pref_k1 + 'data': df['kd_{}_comp_t'.format(query_name)].values,
            pref_k2 + 'data': df['scan_{}_comp_t'.format(query_name)].values,
            'xdata': df['bound'].values,
            'ylabel': 'Time (ms)',
            'xlabel': 'Bound (km)',
            'img_name': 'knn_bound_vary_bound.pdf',
            'title': '{} ( 2 weeks )'.format(query_name_full),
            'ymax': 550,
            'xmargin': 1,
            'ymin': -5,
            'xmin': 0,
            'loc': (0.3, 0.5)
        })
        plot_of(2)

    ###################
    # Ball point vary r
    ###################
    def plot_ball_poit_vary_radius(query_name):
        df = pd.read_csv(os.path.join(DATA_FOLDER_NAME, query_name + '_vary_r.csv'))
        df = df[:30].iloc[::2, :]
        d.update({
            pref_k1 + 'data': df['kd_{}_comp_t'.format(query_name)].values,
            pref_k2 + 'data': df['scan_{}(tm)_comp_t'.format(query_name)].values,
            pref_k3+ 'data': df['scan_{}(sp)_comp_t'.format(query_name)].values,
            'xdata': df['r'].values,
            'ylabel': 'Time (ms)',
            'xlabel': 'Radius (km)',
            'img_name': 'ball_point_vary_radius.pdf',
            'title': '{} ( 2 weeks )'.format(query_name_full),
            'ymax': 250,
            'xmargin': 2,
            'ymin': -5,
            'xmin': -1,
            'loc': 'center right'
        })
        plot_of(3)

    set_up(query_name)
    plot_block_size(query_name)
    plot_spatiotemporal(query_name, bs=40)
    plot_spatiotemporal(query_name, bs=100)
    plot_growing(query_name)
    if query_name == 'knn':
        plot_knn_vary_k('knn')
    elif query_name == 'knn_bound':
        plot_knn_bound_vary_bound('knn_bound')
    elif query_name == 'query_ball':
        plot_ball_poit_vary_radius('query_ball')


######
# MAIN
######
plot_data(query_name='range')
plot_data(query_name = 'knn')
plot_data(query_name = 'knn_bound')
plot_data(query_name='query_ball')
