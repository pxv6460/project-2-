import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore")

class Preprocess:
    def __init__(self, raw_data_path):
        self.df = pd.read_csv(raw_data_path, index_col=0)
        self.df[["Vmag", "Plx", "e_Plx", "B-V"]] = self.df[["Vmag", "Plx", "e_Plx", "B-V"]].apply(pd.to_numeric, errors="coerce")
        self.df = self.df.dropna()

        self.df["Plx"] = self.df["Plx"] / 1000
        self.df["Distance (parsecs)"] = 1/self.df["Plx"]
        self.df["Distance (light years)"] = abs(self.df["Distance (parsecs)"]) * 3.26156
        self.df["Amag"] = self.df["Vmag"] + 5 * (np.log10(self.df["Plx"]) + 1)
        self.df["Temperature (K)"] = 4600 * (1/(0.92*self.df["B-V"] + 1.7) + 1/(0.92*self.df["B-V"] + 0.62))
        self.df["Luminosity (Sun=1)"] = 10**(0.4 * (4.85-self.df["Amag"]))
        self.df["Radius (Sun=1)"] = np.sqrt(self.df["Luminosity (Sun=1)"]) * (5778 / self.df["Temperature (K)"])**2

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna()

        self.df["target"] = self.df["SpType"].apply(self.star_type)
        self.df = self.df.dropna()

    def star_type(self, spectral_type):
        if "VI" in spectral_type:
            return 0
        elif "IV" in spectral_type:
            return 1 # Subgiant
        elif "V" in spectral_type:
            return 0 # Main sequence
        elif "III" in spectral_type or "II" in spectral_type or "Ib" in spectral_type or "Ia" in spectral_type or spectral_type[-1] == 0:
            return 1 # Giant
        elif "M" in spectral_type:
            return 0 # Brown dwarf
        else:
            return None
        
    def get_processed_df(self, numerical=False):
        if(numerical):
            numerical_df = self.df.drop(columns="SpType")
            return numerical_df
        return self.df
    
    def get_df_without(self, columns):
        return self.df.drop(columns=columns)
    
    def corr_heatmap(self):
        plt.figure(figsize=(16, 6))
        sns.heatmap(self.get_processed_df(True).corr(), annot=True)

    def save_csv(self, path):
        self.df.to_csv(path, index=False)
