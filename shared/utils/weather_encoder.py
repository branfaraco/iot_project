import os
import pandas as pd
import numpy as np

class WeatherEncoder:
    """
    Loads cleaned weather data (train/val/test already split),
    computes mean/std from TRAIN ONLY,
    and provides normalized weather vectors.
    """

    def __init__(self, root_cleaned):
        """
        root_cleaned = ".../weather_berlin-tempel/cleaned"
        """
        self.root = root_cleaned

        # Load TRAIN data only for statistics
        self.p_train = pd.read_csv(os.path.join(root_cleaned, "train", "precip.csv"))
        self.t_train = pd.read_csv(os.path.join(root_cleaned, "train", "temp.csv"))
        self.w_train = pd.read_csv(os.path.join(root_cleaned, "train", "wind.csv"))

        for df in (self.p_train, self.t_train, self.w_train):
            df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"])
            drop_cols = [c for c in df.columns if c.strip() in ("STATIONS_ID", "QN")]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)

        # Merge all into a single dataframe indexed by timestamp
        self.train = (
            self.p_train.set_index("MESS_DATUM")
            .join(self.t_train.set_index("MESS_DATUM"), how="inner")
            .join(self.w_train.set_index("MESS_DATUM"), how="inner")
        )

        # Variables to encode (drop QN)
        self.vars = [c for c in self.train.columns if c != "QN"]

        # Replace missing values -999 → NaN
        self.train = self.train.replace(-999, np.nan)

        # Compute normalization stats
        self.mean = self.train[self.vars].mean()
        self.std  = self.train[self.vars].std().replace(0, 1.0)

    def encode_timestamp(self, ts):
        """
        ts = pandas.Timestamp
        Returns a normalized vector (float32).
        """
        row = (
            self.train.reindex([ts])
            .iloc[0:1][self.vars]
            .replace(-999, np.nan)
        )

        # Fill missing → mean
        row = row.fillna(self.mean)

        # Normalize
        norm = (row - self.mean) / self.std

        return norm.values.astype(np.float32).flatten()

    def encode_series(self, df):
        """
        df has MESS_DATUM + weather columns.
        Returns an array [T, D]
        """
        df = df.copy()
        df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"])
        df = df.set_index("MESS_DATUM")[self.vars].replace(-999, np.nan)
        df = df.fillna(self.mean)
        df = (df - self.mean) / self.std
        return df.values.astype(np.float32)
