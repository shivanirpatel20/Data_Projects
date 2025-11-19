import numpy as np
from scipy.io import loadmat
from scipy.spatial import distance
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class OrderOfFaces:
    """
    This class handles loading and processing facial image data for dimensionality
    reduction using the ISOMAP algorithm, with PCA as an optional comparison.

    Attributes:
    ----------
    images_path : str
        Path to the .mat file containing the image dataset.

    Methods:
    -------
    get_adjacency_matrix(epsilon):
        Returns the adjacency matrix based on a given epsilon neighborhood.

    get_best_epsilon():
        Returns the best epsilon for the ISOMAP algorithm, likely based on
        graph connectivity or reconstruction error.

    isomap(epsilon):
        Computes a 2D embedding of the data using the ISOMAP algorithm.

    pca(num_dim):
        Returns a low-dimensional embedding of the data using PCA.
    """

    def __init__(self, images_path='data/isomap.mat'):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """
        image_matrix = loadmat(images_path)
        X_matrix = image_matrix["images"]
        X_matrix = X_matrix.astype(np.float64)

        #the dimensions were not matching up with the question so I decided to switch it
        if X_matrix.shape[0] == 4096 and X_matrix.shape[1] != 4096:
            X_matrix = X_matrix.T

        self.X_matrix = X_matrix
        self.m, self.d = X_matrix.shape

        self.Distance_mat = pairwise_distances(X_matrix) #getting distances of each obs to another obs


        #raise NotImplementedError("Not Implemented")

    def get_adjacency_matrix(self, epsilon: float) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius within which points are considered connected.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """
        Adj_matrix = np.zeros_like(self.Distance_mat)

        for i in range(self.m):
            for j in range(i+1, self.m):
                if self.Distance_mat[i, j] <= epsilon:
                    Adj_matrix[i, j] = self.Distance_mat[i,j]
                    Adj_matrix[j, i] = self.Distance_mat[i, j] #make Adjacency matrix symmetric for all obs
        np.fill_diagonal(Adj_matrix, 0) #zero distances for self distances for each observation
        return Adj_matrix


        #raise NotImplementedError("Not Implemented")

    def get_best_epsilon(self) -> float:
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.

        Returns:
        -------
        float
            Optimal epsilon value ensuring a well-connected neighborhood graph.
        """

        eps_canidates = np.linspace(2, 60, 40) #canidates for the best epsilon (got 15 with 5,50,10 and 12.8 with 4,60,20)
        best_epsilon = 0
        best_fraction = 0

        for eps in eps_canidates:
            Adj_matrix_temp = self.get_adjacency_matrix(eps) #finding A matrix for each epsilon
            G_arr = nx.from_numpy_array(Adj_matrix_temp)
            components = list(nx.connected_components(G_arr)) #getting the components

            if not components:
                continue

            comp_largest = max(components, key=len) #getting the max component in terms of length
            fraction = len(comp_largest) / self.m #equations from slides
            if fraction > best_fraction:
                best_fraction = fraction
                best_epsilon = eps #update the best epsilon when the current fraction is greater than the best fraction. Get largest fraction.
        return best_epsilon
        #raise NotImplementedError("Not Implemented")
    


    def isomap(self, epsilon: float) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius for building the adjacency graph.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.
        """

        Adj_matrix = self.get_adjacency_matrix(epsilon) #get adj matrix from the best epsilon
        
        #computing the distance geo matrix using the shortest path function
        D_geo = shortest_path(Adj_matrix, method="D", directed=False)

        m=D_geo.shape[0]

        H = np.eye(m) - np.ones((m,m)) / m
        D_geo_2 = D_geo ** 2
        C = -.5*H.dot(D_geo_2).dot(H) #THE MDS alg from slides

        m = D_geo.shape[0]

        eigen_vals, eigen_vecs = np.linalg.eigh(C)
        index = np.argsort(eigen_vals)[::-1]

        eigen_vals = eigen_vals[index]
        eigen_vecs = eigen_vecs[:,index]

        Z = eigen_vecs[:,:2].dot(np.diag(np.sqrt(eigen_vals[:2])))

        return Z
        #raise NotImplementedError("Not Implemented")
    


    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.
        """

        pca_alg = PCA(n_components=num_dim) #getting top 2 compnents
        return pca_alg.fit_transform(self.X_matrix) #applying to the data
    
        raise NotImplementedError("Not Implemented")
    
def main():
    #images_path = "/Users/shivanipatel/Downloads/ISYE6740_Fall_2025_HW2-v2-2/gradescope-starter/data/isomap.mat"
    file_path_dir = os.path.dirname(os.path.abspath(__file__))
    images_path = os.path.join(file_path_dir, "data", "isomap.mat")




    order_of_faces = OrderOfFaces(images_path)

    #getting the best epsilon printed:
    best_epsilon = order_of_faces.get_best_epsilon()
    print("best epsilon:", best_epsilon)

    #runing the isomap algorithim:
    Z_matrix = order_of_faces.isomap(best_epsilon) #using best epsilon finding the m by 2 compressed version of X
    X_matrix = order_of_faces.X_matrix
    m = X_matrix.shape[0] #total observations

    #plotting the iso results
    plt.figure(figsize=(8,8))
    plt.scatter(Z_matrix[:,0],Z_matrix[:,1], s=10, alpha=.6) #ask what does alpha do

    #putting sample faces ontop of scatterplot
    sample_face_nodes = [0, m//6, m//4, m//3, m//2, 3*m//4, m-1]
    for face_node in sample_face_nodes:
        x_dim, y_dim = Z_matrix[face_node]
        img = X_matrix[face_node].reshape(64,64)
        im = OffsetImage(img, cmap='gray', zoom=.5)
        ab = AnnotationBbox(im, (x_dim, y_dim), frameon=False)
        plt.gca().add_artist(ab)
    
    plt.title("Embedding using IsoMap Algorithim")
    plt.xlabel("1 Component")
    plt.ylabel("2 Component")
    plt.show()

    #doing the pca algorithim now
    pca_alg = PCA(n_components=2)
    Z_matrix_pca = pca_alg.fit_transform(X_matrix)

    plt.figure(figsize=(8,8))
    plt.scatter(Z_matrix_pca[:,0],Z_matrix_pca[:,1], s=10, alpha=.6)

    for face_node in sample_face_nodes:
        x_dim, y_dim = Z_matrix_pca[face_node]
        img = X_matrix[face_node].reshape(64,64)
        im = OffsetImage(img, cmap='gray', zoom=.5)
        ab = AnnotationBbox(im, (x_dim, y_dim), frameon=False)
        plt.gca().add_artist(ab)

    plt.title("Embedding using PCA Projection Algorithim")
    plt.xlabel("1 PCA Component")
    plt.ylabel("2 PCA Component")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
    
