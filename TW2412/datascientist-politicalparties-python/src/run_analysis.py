
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.visualization import scatter_plot, plot_density_estimation_results, plot_finnish_parties
from political_party_analysis.estimator import DensityEstimator


if __name__ == "__main__":

    data_loader = DataLoader()
    # Data pre-processing step
    ##### YOUR CODE GOES HERE #####
    df_original= data_loader.party_data
    
    df= data_loader.preprocess_data(df_original,
                                    non_features= None,
                                    index= ["party_id", "party", "country"])
  
    
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
        orientation_score= comp_orientation(df, df_with_related_cols.columns)
        orientation= encode_continuous_to_classes(orientation_score, bins, labels= labels)
        df= df.drop(df_with_related_cols.columns, axis=1)
        return orientation, df, orientation_score
    
    orientation, df, orientation_score = intensity_computation(df, bins= None, labels= None)

    # party_ideology= PartyIdeology(df)
    # orientation, df= party_ideology.compute_score()
    # Dimensionality reduction step
    ##### YOUR CODE GOES HERE #####
    dimensionality_red= DimensionalityReducer(data= df, model= None, n_components= 2)
    reduced_dim_data = dimensionality_red.reduce_to_2d()
    original_space= dimensionality_red.map_to_original_space(reduced_dim_data)

    density_estimator= DensityEstimator(reduced_dim_data)
    kde_model = density_estimator.model_distribution(bandwidth= 0.1)
# Step 2: Randomly sample 10 parties
    sampled_parties = density_estimator.sample_parties(kde_model, n_samples=10)

    recovered_parties= dimensionality_red.map_to_original_space(sampled_parties)
    # df_with_related_cols= df_original.loc[:, df_original.columns.str.startswith('l')]


    def heatmap_viz(df: pd.DataFrame):
        df_num= df.select_dtypes(include=['number'])
        plt.figure(figsize=(15, 6))
        sns.heatmap(df_num, cmap="coolwarm", annot=False)
        plt.title("Heatmap of 52 Columns")
        plt.xlabel("Columns")
        plt.ylabel("Samples")
        plt.show()

    heatmap_viz(df)
    heatmap_viz(pd.DataFrame(recovered_parties, columns= df.columns))


    ## Uncomment this snippet to plot dim reduced data
    plt.figure()
    splot = plt.subplot()
    scatter_plot(
        pd.DataFrame(reduced_dim_data),
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    
    # plt.savefig(Path("./plots/dim_reduced_data.png"))
    
    # Density estimation/distribution modelling step
    ##### YOUR CODE GOES HERE #####
    # Plot density estimation results here
    X= pd.DataFrame(reduced_dim_data, columns=['PC1', 'PC2'])
    Y_= np.array(orientation.values)
    valid_indices = ~pd.isna(Y_)
    X = X[valid_indices]
    Y_ = Y_[valid_indices]
    title= 'Estimated density'

# Compute means and covariances for each class
    means = []
    covariances = []

    for cls in np.unique(Y_):
        subset = X[Y_ == cls]
        means.append(subset.mean().values)
        covariances.append(subset.cov().values)

    means = np.array(means)  # Convert to numpy array
    covariances = np.array(covariances)  # Convert to numpy array


    plot_density_estimation_results(X= X,
                                    Y_= Y_,
                                    means= means,
                                    covariances= covariances,
                                    title= title)

    # pyplot.savefig(Path("./plots/Distribution of party orientation.png"))
    
    # Plot left and right wing parties here
    plt.figure()
    splot = plt.subplot()
    ##### YOUR CODE GOES HERE #####
    scatter_plot(
        pd.DataFrame({'Pol orientation': np.array(orientation_score), 'Column2': reduced_dim_data[:, 0]}),
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    plt.title("Lefty/righty parties")
    # pyplot.savefig(Path("./plots/Lefty-righty parties.png"))

    # Plot finnish parties here
    ##### YOUR CODE GOES HERE #####
    df_fin= pd.DataFrame(data= orientation_score)
    df_original.set_index(["party_id", "party", "country"], inplace=True)
    df_fin.index= df_original.index
    plot_finnish_parties(df_fin)  
    # pyplot.savefig(Path("./plots/Finnish parties.png"))
    
    print("Analysis Complete")


