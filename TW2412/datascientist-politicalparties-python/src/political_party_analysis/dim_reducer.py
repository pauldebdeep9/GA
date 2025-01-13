import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, model: any= None, n_components: int = 2):
        
        self.data = data.iloc[:, :].values
        self.original_indices = data.index
        self.feature_columns = data.columns
        self.n_components = n_components

        if model is None:
            self.model = PCA(n_components=n_components)
        elif model == "NMF":
            self.model = NMF(n_components=n_components)
        else:
            self.model = model


    ##### YOUR CODE GOES HERE #####
    def reduce_to_2d(self):
         if hasattr(self.model, "fit_transform"):
            data_2d = self.model.fit_transform(self.data)
            # return without the indices
            return data_2d
            # return pd.DataFrame(data_2d, index=self.original_indices, columns=["Dim1", "Dim2"])
         else:
            raise ValueError("The provided model does not support `fit_transform`.")
    
    def map_to_original_space(self, reduced_samples):
        if hasattr(self.model, 'inverse_transform'):
            original_space = self.model.inverse_transform(reduced_samples)
            return original_space
            # return pd.DataFrame(original_space, index=self.original_indices, columns=self.feature_columns)
        else:
            raise ValueError("The model does not have the `inverse_transform` method.")

if __name__ == "__main__":
    # Generate sample high-dimensional data (e.g., 100 samples with 10 features)
    high_dim_data = np.random.rand(7, 7)
    high_dim_data= pd.DataFrame(data= high_dim_data)
    df = pd.DataFrame(high_dim_data)
    # Reduce the data to 2D
    dimensionality_red= DimensionalityReducer(data= df, model= None, n_components= 2)
    reduced_data = dimensionality_red.reduce_to_2d()
    trans_back=  dimensionality_red.map_to_original_space(reduced_data)
    # Print the result
    print("Reduced Data:", reduced_data)