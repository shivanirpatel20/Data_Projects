from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import os

##pixels: the input image representation. Each row contains one data point (pixel). For image dataset, 
#it contains 3 columns, each column corresponding to Red, Green, and Blue components. 
#Each component has an integer value between 0 and 255.

class KMeansImpl:
    def __init__(self,  max_iterations=100, tolerance=1e-8):

        ## TODO add any params to be needed for the clustering algorithm.
        self.max_iter = max_iterations
        self.tol = tolerance
        pass
        

    def load_image(self, image_name="1.jpeg"):
        """
        Returns the image numpy array.
        It is important that image_name parameter defaults to the choice image name.
        """
        return np.array(Image.open(image_name))

    def compress(self, pixels, num_clusters, norm_distance=2):
        """
        Compress the image using K-Means clustering.

        Parameters:
            pixels: 3D image for each channel (a, b, 3), values range from 0 to 255.
            num_clusters: Number of clusters (k) to use for compression.
            norm_distance: Type of distance metric to use for clustering.
                            Can be 1 for Manhattan distance or 2 for Euclidean distance.
                            Default is 2 (Euclidean).

        Returns:
            Dictionary containing:
                "class": Cluster assignments for each pixel.
                "centroid": Locations of the cluster centroids.
                "img": Compressed image with each pixel assigned to its closest cluster.
                "number_of_iterations": total iterations taken by algorithm
                "time_taken": time taken by the compression algorithm
        """
        map = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {}
        }

        '''
        '''

        #Use k-means with squared-l2 norm as a metric

        #Run k-means implementation with these pictures, with several different k = 3, 6, 12, 24, 48

        #segment the image into k regions in the RGB color space
        #For each pixel in the input image, the algorithm returns a label corresponding to a cluster.

        start = time.time()

        height, width, chan = pixels.shape
        
        #if chan > 3: #forcing chan to be 3 (don't really know why I got this error but this was my fix)
            #pixels = pixels[:, :, :3]
            #height, width, chan = pixels.shape
        image_2d = pixels.reshape(-1, chan).astype(float)
        m = image_2d.shape[0]
        
        np.random.seed(3190)
        random_index = np.random.choice(m, num_clusters, replace=False )
        random_centroids = image_2d[random_index, :]
        
        for iters in range(self.max_iter):
             
             
             #need to make sure the centroids are in an array form
             #centroids_arr = np.asanyarray(random_centroids, dtype=float).reshape(num_clusters, 3)
             centroid_arr = random_centroids.copy()

             if norm_distance == 2:
                 #l2 distance compute
                 c2 = np.sum(centroid_arr ** 2, axis=1)
                 temp_dist = 2 * np.dot(image_2d, random_centroids.T) - c2[None, :]
                 cluster_class = np.argmax(temp_dist, axis=1)
                 #sparse matrix
                 P = csc_matrix((np.ones(m), (np.arange(m), cluster_class)), shape=(m, num_clusters))
                 count = np.array(P.sum(axis=0)).flatten() #this is for l2, comment this out if want to use L1

                 #updating the centroids for each cluster
                 centroids_new = np.zeros_like(random_centroids)
                 for k in range(num_clusters):
                     if count[k] > 0:
                         centroids_new[k] = np.median(image_2d[cluster_class == k], axis=0)
                     else:
                         centroids_new[k] = random_centroids[k]
             #centroids_arr = centroids_new #reassigning the centroid array with new centroids
             elif norm_distance == 1:
                dist = np.zeros((m, num_clusters))
                for k in range(num_clusters):
                    dist[:, k] = np.sum(np.abs(image_2d-random_centroids[k]), axis=1)
                cluster_class = np.argmin(dist, axis=1)

                centroids_new = np.zeros_like(random_centroids)
                for k in range(num_clusters):
                    cluster_check = image_2d[cluster_class == k]
                    if cluster_check.shape[0] > 0:
                        centroids_new[k]=np.median(cluster_check, axis=0)
                    else:
                        centroids_new[k]=random_centroids[k]
             centroid_movement = np.linalg.norm(centroids_new-centroid_arr, ord=1)
             if centroid_movement < self.tol:
                 random_centroids = centroids_new
                 print(f"algorithim converged at {iters+1} iteration")
                 break
             random_centroids = centroids_new

        #reconstructing the image into compressed form
        compressed_image_2d = random_centroids[cluster_class].astype(np.uint8)
        compressed_image_3d = compressed_image_2d.reshape(height, width, chan)

        end = time.time()

        map = {
            "class": cluster_class.reshape(height, width), 
            "centroid": random_centroids,
            "img": compressed_image_3d,
            "number_of_iterations": iters+1,
            "time_taken": end-start,
            "additional_args":{}
        }

        return map

if __name__ == '__main__':
    kmeans_alg = KMeansImpl()
    script_path = os.path.dirname(__file__)
    image_path_parrots = os.path.join(script_path, "data", "parrots.png")
    #image_path_football = os.path.join(script_path, "data", "football.bmp")
    #image_path_og = os.path.join(script_path, "data", "original_image.png")


    image= kmeans_alg.load_image(image_path_parrots)
    #image= kmeans_alg.load_image(image_path_football)
    #image= kmeans_alg.load_image(image_path_og)

    #image = kmeans_alg.load_image("/Users/shivanipatel/Desktop/Screenshot 2025-09-01 at 6.43.35 PM.png")

    for k in [3, 6, 12, 24, 48]:
        final_metrics = kmeans_alg.compress(image, num_clusters=k, norm_distance=2)

        Image.fromarray(final_metrics["img"]).show()
        print(f"k={k}: {final_metrics['number_of_iterations']}, {final_metrics['time_taken']: .2f}s")