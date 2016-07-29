import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt

__author__ = 'Nielsen Rechia'


def main():
    path = 'moto_march/use/'
    filename = ''
    print 'Loading dataset...'
    dataset_read = pd.read_csv(path+filename+'.csv', nrows=None, header=0, index_col=None)
    dataset = dataset_read.values
    print 'Getting n_objects and n_attributes'
    # get n_objects and n_attributes
    n_objects, n_attributes, = dataset.shape
    print str(n_objects) + ' ' + str(n_attributes) + " ..."
    # max number of clusters
    max_nc = int(sqrt(n_objects + 1))
    clusters = range(2, max_nc + 1)
    print 'Clustering from 2 to '+str(max_nc)+' ...'
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

        dbi = []
        dunn = []
        sse = []

        for c in xrange(2, max_nc + 1):
            dif = 0.0
            for cluster_i in xrange(nc):
                instances_i = dataset[np.where(all_labels_kmeans[c - 2] == cluster_i)[0]]
                centroid_i = all_centroids_kmeans[c - 2][cluster_i]
                dif += np.sum(np.power((instances_i - centroid_i), 2))
            print 'SSE for cluster' + str(cluster_i) + ' is ' + str(dif) + ' ...'
            sse += [dif]

        import warnings
        warnings.filterwarnings('error')
        print "DUNN (MAX)..."
        for nc in xrange(2, max_nc + 1):
            dn = 0.0
            max_intra = 0.0
            for cluster_i in xrange(nc):
                instances_i = dataset[np.where(all_labels_kmeans[nc - 2] == cluster_i)[0]]
                pairwase_matrix_intra = pairwise_distances(instances_i, n_jobs=1)
                new_max_intra = np.amax(pairwase_matrix_intra)
                if new_max_intra > max_intra:
                    max_intra = new_max_intra
            for cluster_i in xrange(nc):
                instances_i = dataset[np.where(all_labels_kmeans[nc - 2] == cluster_i)[0]]
                for cluster_j in xrange(nc):
                    if cluster_j > cluster_i:
                        instances_j = dataset[np.where(all_labels_kmeans[nc - 2] == cluster_j)[0]]
                        pairwase_matrix_inter = pairwise_distances(instances_i, instances_j, n_jobs=1)
                        min_inter = np.amin(pairwase_matrix_inter)

                        if dn == 0.0:
                            dn = min_inter/max_intra
                        elif min_inter/max_intra < dn:
                            dn = min_inter/max_intra
            print 'DUNN for k = ' + str(nc) + ' is ' + str(dn) + ' ...'
            dunn += [dn]

        print 'DBI (MIN)...'
        for nc in xrange(2, max_nc + 1):
            db = np.zeros((max_nc, max_nc))
            for cluster_i in xrange(nc):
                centroid_i = all_centroids_kmeans[nc - 2][cluster_i].reshape(1, -1)
                instances_i = dataset[np.where(all_labels_kmeans[nc - 2] == cluster_i)[0]]
                total_i = 0.0
                for cluster_j in xrange(nc):
                    if cluster_j > cluster_i:
                        centroid_j = all_centroids_kmeans[nc - 2][cluster_j].reshape(1, -1)
                        instances_j = dataset[np.where(all_labels_kmeans[nc - 2] == cluster_j)[0]]
                        total_j = 0.0

                        d_ij = float(euclidean_distances(centroid_i, centroid_j))

                        for i in instances_i:
                            total_i += float(euclidean_distances(i.reshape(1, -1), centroid_i))
                        total_i *= (1/float(len(instances_i)))

                        for j in instances_j:
                            total_j += float(euclidean_distances(j.reshape(1, -1), centroid_j))
                        total_j *= (1 / float(len(instances_j)))

                        delta = (total_j + total_i)/d_ij

                        db[cluster_i, cluster_j] = delta

                    elif cluster_j < cluster_i:
                        db[cluster_i, cluster_j] = db[cluster_j, cluster_i]

            res = (1./float(len(db)))*np.sum(np.amax(db, axis=1))
            print 'DBI for k = ' + str(nc) + ' is ' + str(res) + ' ...'
            dbi += [res]

        dbi = np.asarray(dbi, dtype=float)
        dunn = np.asarray(dunn, dtype=float)
        min_dbi = np.argsort(dbi)[0]
        min_dbi_two = np.argsort(dbi)[1]
        max_dunn = np.argsort(dunn)[-1]
        max_dunn_two = np.argsort(dunn)[-2]

        ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=3, colspan=3)
        ax3 = plt.subplot2grid((6, 6), (0, 3), rowspan=3, colspan=3)
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
        manager.window.showMaximized()
        plt.savefig(path + filename + '_plot.png')  # save the figure to file

main()