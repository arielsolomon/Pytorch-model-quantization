import numpy as np
import pandas as pd
import os
import glob


def create_df_different_size(root, col):
    for i in range(1,31,1):
        rows = 500+np.random.choice(200, size=31)
        data = np.random.randint(1, 1000, size=(rows[i], 30))
        df = pd.DataFrame(data=data, index=np.arange(rows[i]), columns=col)
        df.to_csv(root+str(i)+'.csv')
        print(df.shape)

def select_concat(file_list, min_size):

    min_row = min_size
    max_row = min_size
    for i, file in enumerate(file_list):
        row_num = pd.read_csv(file).shape[0]
        if min_row>row_num:
            min_row = row_num
            ind = str(i)
        if max_row<row_num:
            max_row = row_num


    return min_row, int(ind), max_row

col = np.arange(30)
root = '/Data/maneuvers/LSTM_Dataloader/test_df_arangment/'
#create_df_different_size(root, col)
file_list = glob.glob(os.path.join(root+'*.csv'))
min_size = pd.read_csv(file_list[0]).shape[0]
min_row, ind, max_row = select_concat(file_list, min_size)
print('\nMinimal df row num: ', min_row, '\nfile_name: ', file_list[ind], '\nMaxRow: ', max_row)

all_df = pd.DataFrame()
for i,file in enumerate(file_list):

    feature = pd.read_csv(file, index_col=False)['3'].reset_index(drop=True)
    all_df = pd.concat([all_df, feature], axis=1).reset_index(drop=True)
    all_df.iloc[:,i] = all_df.iloc[:,i].shift(max_row-feature.shape[0])
dif = max_row-min_row
all_df = all_df.iloc[dif:,:]
all_df = all_df.iloc[:,1:]
print('y')





