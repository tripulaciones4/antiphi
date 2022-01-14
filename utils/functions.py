import pandas as pd
import numpy as np
from scipy.io import arff

def getProcessedDataFrame(filepath):
  dataset = arff.loadarff(filepath)
  df = pd.DataFrame(dataset[0])
  str_df = df.select_dtypes([np.object]) 
  str_df = str_df.stack().str.decode('utf-8').unstack()

  for col in str_df.columns:
    str_df[col] = str_df[col].astype(int)
  return str_df

def convertEncodingToPositive(dataframe):

  mapping = {-1: 2, 0: 0, 1: 1}

  col_map = {}

  for col in dataframe:
    col_map[col] = mapping

  for i in range(dataframe.shape[0]):
    # if (i%100 == 0):
    #   print(i)
    for j in range(dataframe.shape[1]):
      dataframe.loc[i][j] = mapping[dataframe.loc[i][j]]