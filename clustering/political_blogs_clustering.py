import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import os

class PoliticalBlogsClustering:
    def __init__(self, tol=1e-8):
        self.tol = tol

    def find_majority_labels(self, num_clusters=2):
        map = {
            "overall_mismatch_rate": None,
            "mismatch_rates": []
        }

        #file_path_nodes = "/Users/shivanipatel/Downloads/ISYE6740_Fall_2025_HW1/gradescope_starter_code/Political Blogs/nodes.txt"
        #file_path_edges = "/Users/shivanipatel/Downloads/ISYE6740_Fall_2025_HW1/gradescope_starter_code/Political Blogs/edges.txt"
        

        script_path = os.path.dirname(__file__)
        file_path_nodes = os.path.join(script_path, "data", "nodes.txt")
        file_path_edges = os.path.join(script_path, "data", "edges.txt")



        np.random.seed(3190) #setting the random seed


        #creating the list of all the cluster classes, loading the nodes
        cluster_class = [] 

        with open(file_path_nodes, 'r') as node_lines: #removing leading white space and reading line by line
           for line in node_lines:
             part = line.strip().split('\t')
             cluster_class.append(int(part[2]))
        cluster_class = np.array(cluster_class)
        k_total = len(cluster_class) #total classes


        #loading my edges
        edges = np.loadtxt(file_path_edges, dtype=int)
        i = edges[:,0]-1
        j = edges[:,1]-1
        x = np.ones(edges.shape[0])

        #creating the adj matrix
        A = sparse.coo_matrix((x, (i,j)), shape=(k_total,k_total))
        A=A+A.T #symmetric matrix
        A=sparse.csc_matrix(A).todense()
        A=np.array(A)

        #remove any nodes that are all alone
        not_isolated = A.sum(axis=0) > 0 #have at least one connection
        A = A[not_isolated][:,not_isolated]
        cluster_class = cluster_class[not_isolated]
        k_total = A.shape[0] #new to total of clusters

        #getting the L matrix
        D = np.diag(1/np.sqrt(np.sum(A, axis=1)))
        L = D@A@D
        eigen_vals, eigen_vects = np.linalg.eig(L)

        #for k in [2, 5, 10, 30, 50]:
        k = num_clusters
        
        index_ordered = np.argsort(eigen_vals)
        X = eigen_vects[:, index_ordered[-k:]]
        X=X/np.sqrt((X*X).sum(axis=1))[:,None]
            

        #manually doing the Kmeans
        m, d = X.shape
        random_index = np.random.choice(m, k, replace=False)
        random_centroids = X[random_index, :] #getting the ranodm centroid first

        for iters in range(100):
                #using L2 distance to assign new clusters
                

                centroid_norm = np.sum(random_centroids ** 2, axis=1)
                tmpdiff = 2 * np.dot(X, random_centroids.T) - centroid_norm[None, :]

                class_labels = np.argmax(tmpdiff, axis=1)

                P = csc_matrix((np.ones(m), (np.arange(m), class_labels)), shape=(m,k))
                count = np.array(P.sum(axis=0)).flatten()

                #updating the random centroids
                curr_centroids = np.zeros_like(random_centroids)
                for r in range(k):
                    
                    if count[r] > 0:
                        curr_centroids[r] = np.sum(X[class_labels == r], axis=0)/count[r]
                    else:
                        curr_centroids[r] = random_centroids[r]


                #stop the loop if the centroid movement is less than the tolerance
                centroid_distance = np.linalg.norm(curr_centroids - random_centroids, ord=2)
                random_centroids = curr_centroids

                if centroid_distance < self.tol:
                    #print(f"for K={k}: convergence occurs at iteration {iters+1}")
                    break
        cluster_class_labels = class_labels

        mismatch_rates =[]
        mismatch_overal = 0
        #print(f"k={k}:")

        for clust in range(k):
                index = np.where(cluster_class_labels == clust)[0]
                true_cluster_class = cluster_class[index]
                if len(true_cluster_class) == 0:
                    majority_class = 0
                    mismatch = 0.0
                else:
                   count = np.bincount(true_cluster_class)
                   majority_class = np.argmax(count)
                   mismatch = 1-count[majority_class]/len(true_cluster_class)

       
                mismatch_rates.append({
                    "majority_index":int(majority_class),
                    "mismatch_rate":round(mismatch, 2)
                    })
                mismatch_overal += mismatch*len(index)
                #print(f"cluster label: {clust}: majority: ={majority_class}, mismatch ={mismatch:.3f}")


        overall_mismatch_rate = round(mismatch_overal / k_total, 2)
        #print(f"overall mismatch: {overall_mismatch_rate}")

        map={"overall_mismatch_rate": overall_mismatch_rate,
                       "mismatch_rates": mismatch_rates}

        return map
        
if __name__ == "__main__":
    mismatch_clustering_function = PoliticalBlogsClustering(tol=1e-6)

    result_map = mismatch_clustering_function.find_majority_labels(num_clusters=2)

    print(result_map)
        


    
