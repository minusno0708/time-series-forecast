import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

data_dir = "dataset/"

class Dataset:
    def __init__(self, file):
        self.seq_len = 96
        self.scaler = StandardScaler()
        self.data_loader(file)

    def data_loader(self, file):
        # csv読み込み
        df_raw = pd.read_csv(data_dir + file)

        # データのラベルの取得
        cols = list(df_raw.columns)
        cols.remove("date")
        cols.remove("OT")

        df_raw = df_raw[['date'] + cols + ["OT"]]

        # データを分割
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # データの正規化
        train_data = df_data[:num_train]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_data.values)

        # data_stampの作成
        df_stamp = df_raw[["date"]][:num_train]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(pd.to_datetime(df_stamp["date"].values))
        
        self.data_x = data[:num_train]
        self.data_y = data[:num_train]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        seq_x = self.data_x[index:index+self.seq_len]
        seq_y = self.data_y[index+self.seq_len]
        seq_x_mark = self.data_stamp[index:index+self.seq_len]
        seq_y_mark = self.data_stamp[index+self.seq_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len
    
def time_features(dates):
    features = []
    for index in dates:
        features.append([
            index.hour / 23.0 - 0.5,
            index.dayofweek / 6.0 - 0.5,
            (index.day - 1) / 30.0 - 0.5,
            (index.dayofyear - 1) / 365.0 - 0.5
            ])

    return np.array(features)


if __name__ == "__main__":
    dataset = Dataset("exchange_rate.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
   