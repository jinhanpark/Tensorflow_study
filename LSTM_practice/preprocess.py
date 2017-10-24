import pandas as pd
import numpy as np
import os

from config import Config

def _n_proc(n, data_path="./"):
  file_path = os.path.join(data_path, "market_data_simple.csv")
  df = pd.read_csv(file_path, header=1, index_col="Date", verbose=False)
  arr = np.array(df)
  n_proc = 100*(arr[:-n, :] - arr[n:, :])/arr[n:, :]
  n_proc = np.around(n_proc, 2)
  new_df = pd.DataFrame(n_proc, df.index[:-n], df.columns)
  return new_df

def _extract_companies_alive_for(t, df):
  arr = np.array(df)[:t, :]
  new_df = pd.DataFrame(arr, df.index[:t], df.columns)
  new_df = new_df.dropna(axis=1)
  return new_df

def preprocess(config):
  n = config.proc_gap
  t = config.time_length + 1
  data_path = config.data_path
  batch_size = config.batch_size
  test_ratio = config.test_ratio

  train_path = os.path.join(data_path, "proc_train.csv")
  test_path = os.path.join(data_path, "proc_test.csv")
  
  df = _n_proc(n, data_path)
  df = _extract_companies_alive_for(t, df)
  # df.to_csv("%d_proc_%d_days.csv"%(n, t))

  company_num = df.shape[1]
  df = df.iloc[:, np.random.permutation(company_num)]

  batch_num = company_num // batch_size
  batch_cut = batch_num * batch_size
  df = df.iloc[:, :batch_cut]
  
  truncate_num = config.time_length // config.num_steps
  test_cut = int(truncate_num * test_ratio) * config.num_steps
  train_cut = truncate_num * config.num_steps

  test_df = df.iloc[:test_cut+1, :]
  test_df.to_csv(test_path)
  train_df = df.iloc[test_cut:train_cut+1, :]
  train_df.to_csv(train_path)

  print("preprocess done.")

if __name__ == "__main__":
  preprocess(Config)
