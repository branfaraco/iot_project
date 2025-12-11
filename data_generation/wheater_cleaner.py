import os
import json
import pandas as pd

def load_weather(root):
    precip = pd.read_csv(os.path.join(root, "precipitation.csv"))
    temp   = pd.read_csv(os.path.join(root, "temperature.csv"))
    wind   = pd.read_csv(os.path.join(root, "wind.csv"))

    for df in (precip, temp, wind):
        df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"])
   
    drop_cols = [c for c in df.columns if c.strip() in ("STATIONS_ID", "QN")]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    
    return precip, temp, wind


def filenames_to_days(names):
    """Extract YYYY-MM-DD from filenames like '2019-04-17_BERLIN_8ch.h5'."""
    return [pd.to_datetime(name.split("_")[0]) for name in names]


def filter_by_days(df, days):
    mask = df["MESS_DATUM"].dt.normalize().isin(days)
    return df.loc[mask].copy()


def save_df(df, folder, name):
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, f"{name}.csv"), index=False)


def clean_weather(weather_root, splits_file, out_root):
    precip, temp, wind = load_weather(weather_root)

    with open(splits_file, "r") as f:
        splits = json.load(f)

    subsets = {
        "train": filenames_to_days(splits["train"]),
        "val":   filenames_to_days(splits["val"]),
        "test":  filenames_to_days(splits["test"]),
    }

    for subset, days in subsets.items():
        folder = os.path.join(out_root, subset)
        save_df(filter_by_days(precip, days), folder, "precip")
        save_df(filter_by_days(temp,   days), folder, "temp")
        save_df(filter_by_days(wind,   days), folder, "wind")

    print("Weather cleaning complete.")


if __name__ == "__main__":
    clean_weather(
        weather_root=r"C:\Users\user\UPM\Imperial-4año\IoT\Github\hugging_face\weather_berlin-tempel\interest",
        splits_file=r"C:\Users\user\UPM\Imperial-4año\IoT\Github\hugging_face\BERLIN_reduced\splits\splits.json",
        out_root=r"C:\Users\user\UPM\Imperial-4año\IoT\Github\hugging_face\weather_berlin-tempel\cleaned",
    )