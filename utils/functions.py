import pandas as pd
import numpy as np
from scipy.io import arff
import re

def getProcessedDataFrame(filepath):
  '''Get dataframe processed'''
  dataset = arff.loadarff(filepath)
  df = pd.DataFrame(dataset[0])
  str_df = df.select_dtypes([np.object]) 
  str_df = str_df.stack().str.decode('utf-8').unstack()

  for col in str_df.columns:
    str_df[col] = str_df[col].astype(int)
  return str_df

def convertEncodingToPositive(dataframe):
  '''Convert negative numbers to positive numbers'''

  mapping = {-1: 2, 0: 0, 1: 1}

  col_map = {}

  for col in dataframe:
    col_map[col] = mapping

  for i in range(dataframe.shape[0]):
    # if (i%100 == 0):
    #   print(i)
    for j in range(dataframe.shape[1]):
      dataframe.loc[i][j] = mapping[dataframe.loc[i][j]]

def doble_barra(url):
  '''The existence of “//” within the URL path. If returns 1 is legitimate, if returns 2 is phishing'''
  if ((url)[5] == "/" and url[6] == "/") or (url[6] == "/" and url[7] == "/"):
      n=1
  else:
      n=2
  return n

def guion(url):
  '''The existence of "-" whithin the URL path.If returns 1 is legitimate, if returns 2 is phishing'''
  if "-" in url:
      n=2
  else:
      n=1
  return n

def busqueda_ip(url):
  '''The existence of ip whithin the URL path.If returns 1 is legitimate, if returns 2 is phishing'''
  pattern = re.compile(r'https?:\/\/.[\d]{3}.[\d]{2}.[\d]{3}.[\d]{3}') #busqueda de ruta con numeros
  if pattern.findall(url):
      n=2
  else:
      n=1
  return n

def arroba(url):
  '''The existence of "-" whithin the URL path.If returns 1 is legitimate, if returns 2 is phishing'''
  if '@' in url:
      n=2
  else:
      n=1
  return n

def longitud(url):
  '''Long URL'''
  if len(url) < 54 :
    n=0
  elif len(url) >= 54 and len(url) <= 75:
    n=1
  elif len(url) >= 75 :
    n=2
  return n