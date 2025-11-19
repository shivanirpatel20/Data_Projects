
import os
import numpy as np
from skimage import io, transform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# NOTE: This file consists of 2 classes

# 1. EigenFacesResult - This class should not be modified. Gradescope will use the output of run() 
# method in this format.
# 2. EigenFaces - This is class which will implement the eigen faces algorithm and return the results.  
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# NOTE: This class should NOT be modified.
# Gradescope will depend on the structure of this class as defined. 
# -----------------------------------------------------------------------------
class EigenFacesResult:
    """    
    A structured container for storing the results of the EigenFaces computation.

    Attributes
    ----------
    subject_1_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 1.
        A plt.imshow(map['subject_1_eigen_faces'][0]) should display first in a eigen face for subject 1

    subject_2_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 2.
        A plt.imshow(map['subject_2_eigen_faces'][0]) should display first in a eigen face for subject 2

    s11 : float
        Projection residual of subject 1 test image on subject 1 eigenfaces.

    s12 : float
        Projection residual of subject 2 test image on subject 1 eigenfaces.

    s21 : float
        Projection residual of subject 1 test image on subject 2 eigenfaces.

    s22 : float
        Projection residual of subject 2 test image on subject 2 eigenfaces.
    """

    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22
        
# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class EigenFaces:
    """
    This class handles loading facial images for two subjects, computing eigenfaces
    via PCA, and evaluating projection residuals for test images.

    Methods
    -------
    run():
        Computes the eigenfaces for each subject and the projection residuals for test images.
    """

    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Initializes the EigenFaces object and loads all relevant facial images from the specified directory.

        Parameters
        ----------
        images_root_directory : str
            The path to the root directory containing subject images.
        """

        factor = 4 #reduce a picture of size 16-by-16 to 4-by-4
        n_eigenfaces=6 #need to do this for top 6 eigenfaces
        self.folder = images_root_directory

        self.factor = factor
        self.n_components = n_eigenfaces
        self.subjects = ["subject01", "subject02"]

        self.images_training = {}
        self.resize_image_shapes = {}
        for subj in self.subjects:
            images = []
            for filename in os.listdir(self.folder):
                if filename.startswith(subj) and "test" not in filename:
                    image = io.imread(os.path.join(self.folder, filename))
                    image = np.squeeze(image)

                    resize_image = transform.resize(image, (image.shape[0]//self.factor, image.shape[1]//self.factor), anti_aliasing=True)
                    images.append(resize_image.flatten())
            self.images_training[subj]=np.array(images)
            self.resize_image_shapes[subj] = resize_image.shape

        self.testing_files = ["subject01-test.gif", "subject02-test.gif"]
        self.testing_images = []
        
        for filename in self.testing_files:
            image = io.imread(os.path.join(self.folder, filename))
            image = np.squeeze(image)

            resize_image = transform.resize(image, (image.shape[0]//self.factor, image.shape[1]//self.factor), anti_aliasing=True)
            self.testing_images.append(resize_image.flatten())
        self.testing_images=np.array(self.testing_images)
        #raise NotImplementedError("Not Implemented")
        
    def run(self) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
        """

      
        subject1_pca_alg = PCA(n_components=self.n_components)
        subject1_pca_alg.fit(self.images_training["subject01"])

        subject2_pca_alg = PCA(n_components=self.n_components)
        subject2_pca_alg.fit(self.images_training["subject02"])


        #now reshaping the eigenfaces for each subject
        eigenface_subj01 = np.array([subject1_pca_alg.components_[i].reshape(self.resize_image_shapes["subject01"])
                                     for i in range(self.n_components)]) #do the reshaping for each of the 4 components
        
        eigenface_subj02 = np.array([subject2_pca_alg.components_[i].reshape(self.resize_image_shapes["subject02"])
                                     for i in range(self.n_components)])
        
        
        #computing projection residuals for all eigenfaces (sij) 4 scores
        testing_images = self.testing_images
        proj_resid = np.zeros((2,2)) #storing in a 2 by two array for presentation
        eigenface_subj=[subject1_pca_alg.components_,subject2_pca_alg.components_] #getting the info from compoenents
        #eigenface_subj=[subject1_pca_alg,subject2_pca_alg]

        for j, test_image_curr in enumerate(testing_images):
            for i, eigen_face in enumerate(eigenface_subj):

                projection=eigen_face.T@(eigen_face@test_image_curr)
                proj_resid_curr = np.linalg.norm(test_image_curr-projection)**2
                proj_resid[j,i]=proj_resid_curr
        
        projection_residual_s11=proj_resid[0,0]
        projection_residual_s12=proj_resid[0,1]
        projection_residual_s21=proj_resid[1,0]
        projection_residual_s22=proj_resid[1,1]


        return EigenFacesResult(
           subject_1_eigen_faces=eigenface_subj01,
           subject_2_eigen_faces=eigenface_subj02,
           s11=projection_residual_s11,
           s12=projection_residual_s12,
           s21=projection_residual_s21,
           s22=projection_residual_s22 
        )
    
def main():

    #eigenface_run = EigenFaces(images_root_directory="/Users/shivanipatel/Downloads/ISYE6740_Fall_2025_HW2-v2-2/gradescope-starter/data/yalefaces")
    file_path_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(file_path_dir, "data", "yalefaces")

    eigenface_run = EigenFaces(images_root_directory=images_dir)
    




    results = eigenface_run.run()

    #plotting the 6 eigenfaces for subjects
    for subj_name, eigenfaces in zip(["Subject 1", "Subject 2"], [results.subject_1_eigen_faces, results.subject_2_eigen_faces]):
        fig, axes = plt.subplots(1, eigenface_run.n_components, figsize=(12,4))
        for i in range(eigenface_run.n_components):
            axes[i].imshow(eigenfaces[i], cmap="gray")
            axes[i].axis("off")
            axes[i].set_title(f"eigenface{i+1}")
        fig.suptitle(f"{subj_name} top {eigenface_run.n_components} eigenfaces", fontsize=16)
        plt.show()


    print("subject projection residuals:")
    print(f"s11:{results.s11:.3f}, s12:{results.s12:.3f}, s21:{results.s21:.3f}, s22:{results.s22:.3f}")

if __name__ == "__main__":
    main()

