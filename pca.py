import numpy as np

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class PCA:
    """
    Principal Component Analysis (PCA) implementation for dimensionality reduction.

    This class provides a method to compute the principal components of a dataset
    and reduce its dimensionality by projecting the data onto the top components
    that explain the most variance.
    """

    def __init__(self):
        """
        Initializes the PCA class.
        Currently, no parameters are set during initialization.
        """

        self.components = None
        self.mean = None
        self.explained_variance = None
        
    def fit_transform(self, data: np.ndarray, num_dim: int) -> np.ndarray:
        """
        Perform PCA on the given dataset and return the first `num_dim` principal components.

        Parameters:
        ----------
        data : np.ndarray
            A (m, n) array where each row is a data point with n features.
        num_dim : int
            The number of principal components to return.

        Returns:
        -------
        np.ndarray
            A (m, num_dim) array representing the data projected onto the top `num_dim`
            principal components.

        Notes:
        -----
        Steps typically involved:
        - Center the data by subtracting the mean
        - Compute the covariance matrix
        - Compute eigenvalues and eigenvectors
        - Sort eigenvectors by descending eigenvalues
        - Project the data onto the top `num_dim` eigenvectors
        """

        ##Center the data by subtracting the mean
        self.mean = np.mean(data, axis=0)
        centered_data = data - self.mean

        ##Initialize m,n based on the centered data
        m,n = centered_data.shape

        ##Compute the covariance matrix
        cov_matrix = np.dot(centered_data.T, centered_data) / (m)

        ##Find the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        ##Sort the eigenvectors to determine the top eigenvector
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        ##Explicitly normalize eigenvectors
        for i in range(eigenvectors.shape[1]):
            eigenvectors[:,i] = eigenvectors[:,i] / np.linalg.norm(eigenvectors[:,i])

        ##Make sure the correct signs for the eigenvector
        for i in range(eigenvectors.shape[1]):
            max_index = np.argmax(np.abs(eigenvectors[:,i]))
            if(eigenvectors[max_index,i] < 0):
                eigenvectors[:,i] *= -1

        ##Pick the correct eigenvector
        self.components = eigenvectors[:, :num_dim]
        self.explained_variance = eigenvalues[:num_dim]

        ##Return the (m, num_dim)
        final = np.dot(centered_data, self.components)

        ##Normalize each column of the data after debugging
        for i in range(final.shape[1]):
            final[:,i] = final[:,i] / np.linalg.norm(final[:,i])

        return final
    

