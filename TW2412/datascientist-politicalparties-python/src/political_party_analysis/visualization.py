from itertools import cycle
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter_plot(
    transformed_data: pd.DataFrame,
    color: str = "y",
    size: float = 2.0,
    splot: plt.subplot = None,
    label: Optional[List[str]] = None,
):
    """Write a function to generate a 2D scatter plot."""
    if splot is None:
        plt.figure(figsize=(12, 6))
        splot = plt.subplot()
    columns = transformed_data.columns
    splot.scatter(
        transformed_data.loc[:, columns[0]],
        transformed_data.loc[:, columns[1]],
        size,
        c=color,
        label=label,
    )
    splot.set_aspect("auto")
    splot.set_xlabel("1st Component")
    splot.set_ylabel("2nd Component")
    splot.legend()
    # pyplot.show()


def plot_density_estimation_results(
    X: pd.DataFrame,
    Y_: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    title: str):
    """Use this function to plot the estimated distribution"""
   

    color_iter = cycle(["navy", "c", "cornflowerblue", "gold", "darkorange", "g"])
    plt.figure(figsize=(8, 6))
    splot = plt.subplot()

    unique_classes = np.unique(Y_)
    for i, (label, color) in enumerate(zip(unique_classes, color_iter)):
        cluster_points = X[Y_ == label]
        mean = means[i]
        covar = covariances[i]

        # Scatter plot
        splot.scatter(cluster_points['PC1'], cluster_points['PC2'], color=color, label=f"Class {label}", alpha=0.7)

        # Ellipse plot for covariance
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    # pyplot.show()


def plot_finnish_parties(transformed_data: pd.DataFrame, splot: plt.subplot = None):
    """Write a function to plot the following finnish parties on a 2D scatter plot"""
    finnish_parties = [
        {"parties": ["SDP", "VAS", "VIHR"], "country": "fin", "color": "r"},
        {"parties": ["KESK", "KD"], "country": "fin", "color": "g"},
        {"parties": ["KOK", "SFP"], "country": "fin", "color": "b"},
        {"parties": ["PS"], "country": "fin", "color": "k"},
    ]
    ##### YOUR CODE GOES HERE #####
    # If no subplot is provided, create a new one
    if splot is None:
        _, splot = plt.subplots()

    # Loop through each group of Finnish parties
    for group in finnish_parties:
        parties = group["parties"]
        country = group["country"]
        color = group["color"]

        # Filter data for the specified parties and country
        filtered_data = transformed_data.loc[
            (transformed_data.index.get_level_values('party').isin(parties)) &
            (transformed_data.index.get_level_values('country') == country)
        ]

        # Plot the filtered data
        splot.scatter(
            filtered_data.index.get_level_values('party'),  # X-axis: party names
            filtered_data,  # Y-axis: lrgen values
            color=color,
            label=", ".join(parties),
        )

    # Add labels, legend, and title
    splot.set_xlabel("Party")
    splot.set_ylabel("Orientation")
    splot.legend()
    splot.set_title("Scatter Plot of Finnish Parties")

    # Display the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

synth_array= np.random.randn(100, 2)

def trans_df(X: np.array) -> pd.DataFrame:
    return pd.DataFrame(data= X)

if __name__ == '__main__':
    pass


