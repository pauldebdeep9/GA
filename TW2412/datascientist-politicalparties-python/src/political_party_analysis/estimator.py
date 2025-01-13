import pandas as pd
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """
    # def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.kde_model = None
        self.feature_names = None

    ##### YOUR CODE GOES HERE #####
# Question 1: Write a function to model the distribution of the political party dataset
    def model_distribution(self, bandwidth=1.0):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(self.data)
        self.kde_model = kde
# Question 2: Write a function to randomly sample 10 parties from this distribution
    def sample_parties(self, kde_model, n_samples=10):
       if self.kde_model is None:
            raise ValueError("KDE model not fitted. Call `model_distribution` first.")
       samples = self.kde_model.sample(n_samples)
       return samples

# Question 3: Map the randomly sampled 10 parties back to the original higher dimensional space
    def map_to_original_space(reduced_samples, pca_model):
        if hasattr(pca_model, 'inverse_transform'):
            original_space = pca_model.inverse_transform(reduced_samples)
            return original_space
        else:
            raise ValueError("PCA model does not have the `inverse_transform` method.")
    

def encode_continuous_to_classes(series, bins, labels=None):
    if labels is None:
        labels = range(len(bins) - 1)
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

def comp_orientation(df, col_names):
        orientation= 0.5*(df[col_names[0]] + df[col_names[1]]) - df[col_names[2]]
        return orientation

def intensity_computation(df, bins= None, labels= None):
    df_with_related_cols= df.loc[:, df.columns.str.startswith('l')]
    bins = [1, 4, 7, 10] 
    labels = [1, 2, 3]
    political_orient= comp_orientation(df, df_with_related_cols.columns)
    orientation= encode_continuous_to_classes(political_orient, bins, labels= labels)
    df= df.drop(df_with_related_cols.columns, axis=1)
    return orientation, df



class PartyIdeology:
    def __init__(self,
                 df):
        self.df= df

    def encoding(self, series, bins, labels=None):
        if labels is None:
            labels = range(len(bins) - 1)
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)
   
    def compute_orientation(self, col_names):
        orientation= 0.5*(self.df[col_names[0]] + self.df[col_names[1]]) - self.df[col_names[2]]
        return orientation

    def compute_score(self, bins= None, labels= None):
        df_with_related_cols= self.df.loc[:, self.df.columns.str.startswith('l')]
        bins = bins or [1, 4, 7, 10] 
        labels = labels or [1, 2, 3]
        political_orient= self.compute_orientation(df_with_related_cols.columns)
        orientation= self.encoding(political_orient, bins, labels)
        self.df= self.df.drop(df_with_related_cols.columns, axis=1)
        return orientation, self.df
    

# Example Usage
if __name__ == "__main__":
    # Assuming you have the reduced data and PCA model from dimensionality reduction
    reduced_data = np.random.rand(100, 2)  # Example reduced data (replace with actual data)
    density_estimator= DensityEstimator(reduced_data)
    # Step 1: Model the distribution
    kde_model = density_estimator.model_distribution(bandwidth= 1.0)
    # Step 2: Randomly sample 10 parties
    sampled_parties = density_estimator.sample_parties(density_estimator.kde_model, n_samples=10)
    print("Sampled Parties (Reduced Space):\n", sampled_parties)