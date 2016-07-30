from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

__author__ = 'Nielsen Rechia'

"""
    Test
"""


class Measurements(object):
    def __init__(self):
        pass

    @staticmethod
    def dbi(max_nc, all_centroids, all_labels, dataset):
        dbi = []
        print 'DBI (MIN)...'
        for nc in xrange(2, max_nc + 1):
            db = np.zeros((max_nc, max_nc))
            for cluster_i in xrange(nc):
                centroid_i = all_centroids[nc - 2][cluster_i].reshape(1, -1)
                instances_i = dataset[np.where(all_labels[nc - 2] == cluster_i)[0]]
                total_i = 0.0
                for cluster_j in xrange(nc):
                    if cluster_j > cluster_i:
                        centroid_j = all_centroids[nc - 2][cluster_j].reshape(1, -1)
                        instances_j = dataset[np.where(all_labels[nc - 2] == cluster_j)[0]]
                        total_j = 0.0

                        d_ij = float(euclidean_distances(centroid_i, centroid_j))

                        for i in instances_i:
                            total_i += float(euclidean_distances(i.reshape(1, -1), centroid_i))
                        total_i *= (1 / float(len(instances_i)))

                        for j in instances_j:
                            total_j += float(euclidean_distances(j.reshape(1, -1), centroid_j))
                        total_j *= (1 / float(len(instances_j)))

                        delta = (total_j + total_i) / d_ij

                        db[cluster_i, cluster_j] = delta

                    elif cluster_j < cluster_i:
                        db[cluster_i, cluster_j] = db[cluster_j, cluster_i]

            res = (1. / float(len(db))) * np.sum(np.amax(db, axis=1))
            print 'DBI for k = ' + str(nc) + ' is ' + str(res) + ' ...'
            dbi += [res]
        return dbi

    @staticmethod
    def dunn(max_nc, all_labels, dataset):
        dunn = []
        print "DUNN (MAX)..."
        for nc in xrange(2, max_nc + 1):
            dn = 0.0
            max_intra = 0.0
            for cluster_i in xrange(nc):
                instances_i = dataset[np.where(all_labels[nc - 2] == cluster_i)[0]]
                pairwase_matrix_intra = pairwise_distances(instances_i, n_jobs=1)
                new_max_intra = np.amax(pairwase_matrix_intra)
                if new_max_intra > max_intra:
                    max_intra = new_max_intra
            for cluster_i in xrange(nc):
                instances_i = dataset[np.where(all_labels[nc - 2] == cluster_i)[0]]
                for cluster_j in xrange(nc):
                    if cluster_j > cluster_i:
                        instances_j = dataset[np.where(all_labels[nc - 2] == cluster_j)[0]]
                        pairwase_matrix_inter = pairwise_distances(instances_i, instances_j, n_jobs=1)
                        min_inter = np.amin(pairwase_matrix_inter)

                        if dn == 0.0:
                            dn = min_inter / max_intra
                        elif min_inter / max_intra < dn:
                            dn = min_inter / max_intra
            print 'DUNN for k = ' + str(nc) + ' is ' + str(dn) + ' ...'
            dunn += [dn]
        return dunn

    @staticmethod
    def sse(max_nc, all_labels, all_centroids, dataset):
        sse = []
        for nc in xrange(2, max_nc + 1):
            dif = 0.0
            for cluster_i in xrange(nc):
                instances_i = dataset[np.where(all_labels[nc - 2] == cluster_i)[0]]
                centroid_i = all_centroids[nc - 2][cluster_i]
                dif += np.sum(np.power((instances_i - centroid_i), 2))
            print 'SSE for cluster' + str(cluster_i) + ' is ' + str(dif) + ' ...'
            sse += [dif]
        return sse
