import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
# plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
# colors = pd.tools.plotting._get_standard_colors(10, color_type='random')
colors = range(10)


def heatmap(gridsearch, title="", prefix_filename="heatmap", params_of_interest=None):
    if hasattr(gridsearch, 'param_distributions'):
        if isinstance(gridsearch.param_distributions, dict):
            cv_params_keys = gridsearch.param_distributions.keys()
        else:
            cv_params_keys = list(set(sum([x.keys() for x in gridsearch.param_distributions], [])))
    elif hasattr(gridsearch, 'param_grid'):
        if isinstance(gridsearch.param_grid, dict):
            cv_params_keys = gridsearch.param_grid.keys()
        else:
            cv_params_keys = list(set(sum([x.keys() for x in gridsearch.param_grid], [])))
    else:
        raise Exception("gridsearch should be a GridSearchCV object or a"
                        " RandomizedSearchCV object already fitted from sklearn")

    if not hasattr(gridsearch, 'cv_results_'):
        raise Exception("The model should be trained before calling this function")

    if params_of_interest is None:
        params_of_interest = cv_params_keys

    cv_params_keys = ["param_" + a for a in cv_params_keys if a in params_of_interest]
    # print cv_params_keys
    cv_results = pd.DataFrame(gridsearch.cv_results_)[cv_params_keys + ["mean_test_score", "mean_train_score"]]
    for phase in ['test', 'train']:
        for (a, b) in combinations(cv_params_keys, 2):
            if a > b:
                a, b = b, a
            df = pd.pivot_table(cv_results, index=[a], columns=[b],
                                values=["mean_{}_score".format(phase)], aggfunc=np.mean)
            plt.figure(figsize=(8, 6))
            plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
            plt.imshow(df, interpolation='nearest')
            x_name = str(df.columns.levels[1].name)[6:]
            y_name = str(df.index.name)[6:]
            x_ticks = df.columns.levels[1].values
            if type(x_ticks[0]) == float:
                x_ticks = np.round(x_ticks, 3)
            y_ticks = df.index.values
            if type(y_ticks[0]) == float:
                y_ticks = np.round(y_ticks, 3)
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.colorbar()
            plt.xticks(np.arange(len(x_ticks)), x_ticks, rotation=45)
            plt.yticks(np.arange(len(y_ticks)), y_ticks)
            plt.title(title)

            plt.savefig("{}_{}_{}_{}.png".format(prefix_filename, x_name, y_name, phase))
            plt.close()


def scatter_plot(x, y, show_diag=True, x_label="", y_label="", figname=""):
    plt.figure()
    plt.plot(x, y, 'ko')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show_diag:
        ax = plt.gca()
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        xymin, xymax = min(xmin, ymin), max(xmax, ymax)
        ax.set(xlim=(xymin, xymax), ylim=(xymin, xymax))
        # Plot your initial diagonal line based on the starting
        # xlims and ylims.
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    if figname != "":
        plt.savefig(figname)
    else:
        plt.show()
    plt.clf()
    plt.close()


def timeseries_plot(x, y, groups=None, x_label="", y_label="", legend_title="", title="", figname=""):
    if groups is None:
        groups = [1]*len(x)
    df = pd.DataFrame(dict(x=x, y=y, label=groups))

    df = df.sort_values(['label', 'x', 'y'])

    groups = df.groupby('label')

    fig, ax = plt.subplots()
    ax.set_color_cycle(colors)
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='-', ms=5, label=name)
    ax.legend(numpoints=1, title=legend_title, fontsize=8, loc='best',
          ncol=2, fancybox=True, shadow=True)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    fig.suptitle(title, fontsize=10)
    if figname != "":
        plt.savefig(figname)
    else:
        plt.show()
    plt.clf()
    plt.close()


def box_plot(list_data, list_labels, x_label="", y_label="", figname=""):
    plt.figure()
    plt.boxplot(list_data, labels=list_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if figname != "":
        plt.savefig(figname)
    else:
        plt.show()
    plt.clf()
    plt.close()


def hist_plot(x, groups=None, show_density=True, x_label="", y_label="", legend_title="", title="", figname=""):
    if groups is None:
        groups = [1]*len(x)

    df = pd.DataFrame(dict(x=x, label=groups))
    df = df.sort_values(['label', 'x'])
    groups = df.groupby('label')

    fig = plt.figure()
    ax = fig.gca()
    for (name, group) in groups:
        if show_density:
            group['x'].plot(kind="kde",  label=name)
        group["x"].hist(bins=50, normed=True)

    if figname != "":
        plt.savefig(figname)
    else:
        plt.show()
    plt.clf()
    plt.close()


def motif_plot(peptides, fname, format="PNG",**kwds):
    """
    uses the Berkeley weblogo service to download and save a weblogo of itself

    requires an internet connection.
    The parameters from **kwds are passed directly to the weblogo server.
    """
    from urllib import urlencode
    from urllib2 import urlopen, Request

    url = 'http://weblogo.berkeley.edu/logo.cgi'
    values = {'sequence': "\n".join(peptides),
              'format': format,
              'logowidth': '18',
              'logoheight': '5',
              'logounits': 'cm',
              'kind': 'AUTO',
              'firstnum': "1",
              'uniform': "on",
              'command': 'Create Logo',
              'smallsamplecorrection': "on",
              'symbolsperline': 32,
              'res': '96',
              'res_units': 'ppi',
              'antialias': 'on',
              'title': '',
              'barbits': '',
              'xaxis': 'on',
              'xaxis_label': '',
              'yaxis': 'on',
              'yaxis_label': '',
              'showends': 'on',
              'shrink': '0.5',
              'fineprint': 'on',
              'ticbits': '1',
              'colorscheme': 'DEFAULT',
              'color1': 'green',
              'color2': 'blue',
              'color3': 'red',
              'color4': 'black',
              'color5': 'purple',
              'color6': 'orange',
              'color1': 'black',
              }
    for k, v in kwds.items():
        values[k]=str(v)

    data = urlencode(values)
    req = Request(url, data)
    response = urlopen(req)
    with open(fname, "w") as f:
        im=response.read()
        f.write(im)

if __name__ == '__main__':
    hist_plot([1,1,1,1,1,2,1,2,2,2,23,43,3,33,33,2,3,43,121,331,3,21,12,34,2,23])