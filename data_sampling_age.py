import json
from pandas import json_normalize
import torch
from torch import nn
import sys
import numpy as np
import pandas as pd
import numpy as np
from pprint import pprint 
import gc

from soynlp.normalizer import *
from konlpy.tag import Okt
import re

# json file을 읽어오고 dataframe으로 변환하는 함수
def json2df (file):
  with open(file, 'r', encoding='UTF8') as f:
    jdata = json.load(f)
  
  meta_list = [['header', 'dialogueInfo', 'numberOfParticipants'],
             ['header', 'dialogueInfo', 'numberOfUtterances'],
             ['header', 'dialogueInfo', 'dialogueID'],
            ]
  # dataframe of utterances 
  df1 = pd.json_normalize(jdata['data'][:], record_path=['body'], meta= meta_list, errors='ignore')
  # dataframe of participants info
  df2 = pd.json_normalize(jdata['data'], record_path=['header', 'participantsInfo'],  meta= [['header', 'dialogueInfo', 'dialogueID']], errors='ignore')
  # join df1 and df2 where (df1.dialogueID == df2.dialogueID and df1.participantID == df2.participantID)
  df = pd.merge(left = df1 , right = df2, how = "left", on = ["header.dialogueInfo.dialogueID", 'participantID' ],sort=False)

  del df1
  gc.collect()
  del df2
  gc.collect()

  # rename column names
  # 열이름 너무 길어서 바꿈. P는 참가자, U는 메시지, D는 대화, T는 turn 으로 통일시킴. 맘에 안 들면 바꿔도 ok
  df.rename(columns = {'header.dialogueInfo.numberOfParticipants' : 'P_num', 
                       'header.dialogueInfo.numberOfUtterances': 'U_num', 
                       'header.dialogueInfo.dialogueID':'D_id',
                       'utteranceID':'U_id',
                       'participantID':'P_id',
                       'gender':'P_gender',
                       'age':'P_age'}, inplace = True)

  df= df[['utterance', 'U_id', 'P_id', 'P_num', 'D_id', 'P_age']]

  return df

# input dataframe을 우리의 형식에 맞게 변형(i.e. 화자가 보낸 메시지로 묶어줌)

def clean_df_age(df):
  age12 = ['10대', '20대']
  age34 = ['30대', '40대']
  age567 = ['50대', '60대', '70대']

  df['P_age'] = [0 if age in age12 else (1 if age in age34 else (2 if age in age567 else np.nan)) for age in df['P_age']]
  df = df.set_index(['D_id','P_id'] )
  df = df.sort_index()
  df['index'] = df.index.to_numpy()
  s = df.groupby(df.index)['utterance'].apply(' '.join).to_frame()
  df = pd.merge(left = s.reset_index(), right = df[['P_age','index']], how = "left", left_on=['index'], right_on= ['index'], sort=False).drop_duplicates().set_index('index').reset_index()
  df = df.dropna(axis=0)

  del s
  gc.collect()

  return df


# input dataframe을 주어진 비율로 sample
def sample_df(df, ratio):
  n = int(len(df) * ratio)
  result = df.sample(n=n, random_state=42)

  return result

# data label 비율 맞춰주는 함수
# sample을 먼저 할까, gender 비율을 먼저 맞출까 하다가 큰 상관 없을 거 같아서 sampling 먼저 하고 gender 비율 맞춤


def fix_age_bias(df):
  df_12 = df[df['P_age']==0]
  df_34 = df[df['P_age']==1]
  df_567 = df[df['P_age']==2]


  print()
 
  print("10-20대 수 : {}, 30-40대 수 : {}, 50-70대 수 : {}".format(df_12['P_age'].value_counts()[0], df_34['P_age'].value_counts()[1], df_567['P_age'].value_counts()[2])) 

  df_12 = df_12.sample(n= df_567['P_age'].value_counts()[2], random_state=42)
  df_34 = df_34.sample(n= df_567['P_age'].value_counts()[2], random_state=42)

  df = pd.concat([df_12, df_34, df_567],sort=False)
  print("----After adjusting data imbalance in dataset----")
  print("10-20대 수 : {}, 30-40대 수 : {}, 50-70대 수 : {}".format(df_12['P_age'].value_counts()[0], df_34['P_age'].value_counts()[1], df_567['P_age'].value_counts()[2])) 

  del df_12
  gc.collect() 
  del df_34
  gc.collect()
  del df_567
  gc.collect()


  return df

# 주어진 비율로 json 파일들을 sampling하고 통합하는 함수
# 입력으로 jsonfile 의 string의 list와 sampling 할 ration를 받는다/
def sample_and_merge(jsonfile_list, ratio):

  result = None

  for i, file in enumerate(jsonfile_list):
    df = json2df(file)
    df = clean_df_age(df)
    
    

    print("number of participants in '{}' is {}".format(file, len(df)))
    
    df = sample_df(df, ratio)
    
    print("---- After sampling with ratio {} ----".format(ratio))
    print("number of participants is {}".format(len(df)))

    df= fix_age_bias(df)

    print()
    
    if (i==0):
      result = df.copy()
    else:
      result = pd.concat([result,df])
  
  result = result.set_index("index").reset_index()

  print("Total length of the dataframe is {}".format(len(result)))
  

  return result


def json2csv (jsonfile_list, ratio, output_file):
    df = sample_and_merge(jsonfile_list, ratio)
    result = df.to_csv(output_file, index=False, encoding="utf-8-sig") 

    return result 



json_list = [
            'event.json',
            'beauty_health.json',
            'education.json',
            'food.json',
            'individual.json',
            'leisure.json',
            'living.json',
            'shopping.json',
            'work_job.json'
            ]


json2csv(json_list, 1, 'age_test.csv') 

