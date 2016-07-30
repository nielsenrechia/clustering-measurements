from measurements import Measurements
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from math import sqrt
import numpy as np


if __name__ == '__main__':
    filename = 'iris.csv'
    print 'Loading dataset ...'
    dataset = pd.read_csv(filename, nrows=None, header=0, index_col=None, usecols=['sepal_length', 'sepal_width',
                                                                                   'petal_length', 'petal_width'])
    dataset = dataset.values
    print 'Getting n_objects and n_attributes ...'
    n_objects, n_attributes, = dataset.shape
    print str(n_objects) + ' objects and ' + str(n_attributes) + " attributes..."
    # max number of clusters
    max_nc = int(sqrt(n_objects + 1))
    clusters = range(2, max_nc + 1)
    print 'Clustering from 2 to ' + str(max_nc) + ' ...'
    # seeds
    np.random.seed(0)
    max_seed = 30

    all_labels_kmeans = []
    all_centroids_kmeans = []
    all_inertia_kmeans = []
    header = []
    for nc in xrange(2, max_nc + 1):
        print 'Kmeans clustering with k = ' + str(nc) + ' ...'
        header += [nc]
        inst = KMeans(n_clusters=nc, max_iter=100000, n_init=max_seed, init='k-means++', n_jobs=1)
        all_labels_kmeans += [inst.fit_predict(dataset)]
        all_centroids_kmeans += [inst.cluster_centers_]
        all_inertia_kmeans += [inst.inertia_]

    print 'Clustering Measurements for Kmeans ...'

    metrics = Measurements()
    dunn = metrics.dunn(max_nc, all_labels_kmeans, dataset)
    dbi = metrics.dbi(max_nc, all_centroids_kmeans, all_labels_kmeans, dataset)
    sse = metrics.sse(max_nc, all_labels_kmeans, all_centroids_kmeans, dataset)

    dbi = np.asarray(dbi, dtype=float)
    dunn = np.asarray(dunn, dtype=float)
    min_dbi = np.argsort(dbi)[0]
    min_dbi_two = np.argsort(dbi)[1]
    max_dunn = np.argsort(dunn)[-1]
    max_dunn_two = np.argsort(dunn)[-2]

    ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=3, colspan=3)
    ax4 = plt.subplot2grid((6, 6), (3, 0), rowspan=3, colspan=3)
    ax6 = plt.subplot2grid((6, 6), (3, 3), rowspan=3, colspan=3)

    ax1.plot(clusters, sse, 'b*-', label='SSE')
    ax1.grid(True)
    ax1.legend(loc='upper right', shadow=True)

    ax4.plot(clusters, dbi, 'gp-', label='DBI')
    ax4.plot(clusters[min_dbi], dbi[min_dbi], marker='o', markersize=12, markeredgewidth=2,
             markeredgecolor='g', markerfacecolor='None')
    ax4.plot(clusters[min_dbi_two], dbi[min_dbi_two], marker='o', markersize=12, markeredgewidth=2,
             markeredgecolor='r', markerfacecolor='None')
    ax4.text(clusters[min_dbi], dbi[min_dbi], '  <- %i' % clusters[min_dbi])
    ax4.text(clusters[min_dbi_two], dbi[min_dbi_two], '  <- %i' % clusters[min_dbi_two])
    ax4.grid(True)
    ax4.legend(loc='upper right', shadow=True)

    ax6.plot(clusters, dunn, 'cd-', label='Dunn Index')
    ax6.plot(clusters[max_dunn], dunn[max_dunn], marker='o', markersize=12, markeredgewidth=2,
             markeredgecolor='g', markerfacecolor='None')
    ax6.plot(clusters[max_dunn_two], dunn[max_dunn_two], marker='o', markersize=12, markeredgewidth=2,
             markeredgecolor='r', markerfacecolor='None')
    ax6.text(clusters[max_dunn], dunn[max_dunn], '  <- %i' % clusters[max_dunn])
    ax6.text(clusters[max_dunn_two], dunn[max_dunn_two], '  <- %i' % clusters[max_dunn_two])
    ax6.grid(True)
    ax6.legend(loc='upper right', shadow=True)

    manager = plt.get_current_fig_manager()
    plt.show()
    # manager.window.showMaximized()
    # plt.savefig(path + filename + '_plot.png')  # save the figure to file