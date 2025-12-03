import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy 

class Hierarchical_Clustering:
    def __init__(self):
        pass
    def rgb_hex(self, color):
        '''converts a (r,g,b) color (either 0-1 or 0-255) to its hex representation.
        for ambiguous pure combinations of 0s and 1s e,g, (0,0,1), (1/1/1) is assumed.'''
        #message='color must be an iterable of length 3.'
        #assert hasattr(color, '__iter__'), message
        #assert len(color)==3, message
        #if all((c <= 1) & (c >= 0) for c in color): color=[int(round(c*255)) for c in color] # in case provided rgb is 0-1
        #color=tuple(color)
        color = np.array(color)
        return matplotlib.colors.to_hex(color)

    def get_cluster_colors(self, n_clusters, alpha=0.8, alpha_outliers=0.05):
        #my_set_of_20_rgb_colors =
        cluster_colors = [
            [
                np.random.randint(255)/255,
                np.random.randint(255)/255,
                np.random.randint(255)/255,
            ]
            for _ in range(1000)
        ]
        cluster_colors = [c+[alpha] for c in cluster_colors]
        outlier_color = [0,0,0,alpha_outliers]
        return [cluster_colors[i%19] for i in range(n_clusters)] + [outlier_color]

    def clusters(self, X, threshold,no_plot = True, method='ward', metric='euclidean', default_color='black'):

        # perform hierarchical clustering
        Z              = hierarchy.linkage(X, method=method, metric=metric)

        # get cluster labels
        labels         = hierarchy.fcluster(Z, threshold, criterion='distance') - 1
        labels_str     = [f"cluster #{l}: n={c}\n" for (l,c) in zip(*np.unique(labels, return_counts=True))]
        n_clusters     = len(labels_str)
        cluster_colors = [self.rgb_hex(c[:-1]) for c in self.get_cluster_colors(n_clusters, alpha=0.8, alpha_outliers=0.05)]
        cluster_colors_array = [cluster_colors[l] for l in labels]
        link_cols = {}
        for i, i12 in enumerate(Z[:,:2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(Z) else cluster_colors_array[x] for x in i12)
            link_cols[i+1+len(Z)] = c1 if c1 == c2 else 'k'

        # plot dendrogram with colored clusters
        if no_plot:
            self._extracted_from_clusters_20(threshold)
        # plot dendrogram based on clustering results
        dend = hierarchy.dendrogram(
            Z,
            no_plot = not no_plot,
            labels = labels,
            color_threshold=threshold,
            truncate_mode = 'level',
            p = 5,
            show_leaf_counts = True,
            leaf_rotation=90,
            leaf_font_size=10,
            show_contracted=False,
            link_color_func=lambda x: link_cols[x],
            above_threshold_color=default_color,
            distance_sort='descending',
            )


        self.labels = labels
        self.dendogram = dend

    # TODO Rename this here and in `clusters`
    def _extracted_from_clusters_20(self, threshold):
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data points')
        plt.ylabel('Distance')
        plt.axhline(threshold, color='k')
        fig.patch.set_facecolor('white')
        #Ticker(ax)
