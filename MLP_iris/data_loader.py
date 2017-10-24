import pandas as pd
import numpy as np

class GetIrisData:
    def __init__(self, data_path):
        df = pd.read_csv(data_path, header=None)
        df = np.array(df)

        #df = self.shuffle_inside_class(df)
        df = self.rearrange_for_test(df)
        df = self.class_name_to_onehot(df)
        df[:, :4] = self.normalize_input_data(df[:, :4])

        sep_ind = 105
        self.split_data(df, sep_ind)

    def shuffle_inside_class(self, df):
        for i in range(3):
            np.random.shuffle(df[i*50:(i+1)*50])
        return df
        
    def rearrange_for_test(self, df):
        df = np.vsplit(df, [35, 50, 85, 100, 135])
        df = np.concatenate((df[0], df[2], df[4], df[1], df[3],  df[5]))
        return df

    def normalize_input_data(self, df):
        df = df.astype(np.float32)
        df -= np.mean(df, axis = 0)
        df /= np.std(df, axis = 0)
        return df
    
    def class_name_to_onehot(self, df):
        classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        df = np.hstack((df, np.zeros((df.shape[0], 2))))
        for i in range(len(classes)):
            onehot = np.zeros(3)
            onehot[i] = 1
            condition = df[:, 4] == classes[i]
            df[:, 4:7][condition] = onehot
        return df
        
    def split_data(self, df, sep_ind):
        df = np.vsplit(df, [sep_ind])
        self.training_data = np.hsplit(df[0], [4])
        self.test_data = np.hsplit(df[1], [4])

