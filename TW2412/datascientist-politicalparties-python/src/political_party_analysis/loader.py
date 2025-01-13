from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path = Path(__file__).parents[2].joinpath("data", "CHES2019V3.dta")
           
    # Check if the file already exists
        if not data_path.exists():
        # Download the file if it does not exist
            urlretrieve(self.data_url, data_path)
    
    # Load the data from the file
        return pd.read_stata(data_path)
    
    
    def remove_duplicates(self, df_original: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        df = df_original.drop_duplicates()
    # print('Shape of df before and after removing duplicates: df_orginal.shape, df.shape')
        print(f"Shape of DataFrame before and after removing duplicates: {df_original.shape}, {df.shape}")
        return df

    def remove_nonfeature_cols(
        self, df_original: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        df= df_original.copy()
        if index:
            df= df.set_index(index)
        if non_features:
            df = df.drop([col for col in non_features if col in df.columns], axis=1)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        df.fillna(df.mean(), inplace=True)
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        ##### YOUR CODE GOES HERE #####
        index= self.party_data.index
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        return pd.DataFrame(scaled_features, columns=df.columns, index= index)

    def preprocess_data(self, df_original, non_features, index):
        """Write a function to combine all pre-processing steps for the dataset"""
        ##### YOUR CODE GOES HERE #####
        df= self.remove_duplicates(df_original)
        df= self.remove_nonfeature_cols(df_original, non_features= None, index= index)
        df= self.handle_NaN_values(df)
        # df= self.scale_features(df)
        return df
    
   
if __name__ == '__main__':
    data_loader= DataLoader()
    df_original= data_loader.party_data
    # df= data_loader.scale_features(df_original)
    df= data_loader.preprocess_data(df_original,
                                    non_features= None,
                                    index= ["party_id", "party", "country"])
    print(df.describe())
