import pandas as pd
import numpy as np


class GetData:
    def __init__(self, data_path):
        df = pd.read_excel(data_path, sheetname=0)
        df = np.array(df.iloc[0:5, 1:7])
        df = self.rescale_without_noreply(df)

        self.data = [df, df[1:, :]/100.]

    def rescale_without_noreply(self, df):
        for i in range(df.shape[0]):
            rescale_factor = (100 - df[i, -1])/100
            df[i, :] = df[i, :]/rescale_factor
        print(df)
        return df[:, :-1]
