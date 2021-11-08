import numpy as np 
import pandas as pd 

src_class = 'NS'
frac = 0.9 

data = pd.read_csv('source_list/'+src_class+'_data_clean_source_list.csv')
data =  data.sample(frac=1)
print(data.describe())
frac_no = int(len(data)*frac)
train_data = data.iloc[:frac_no]
test_data = data.iloc[frac_no:]
print(train_data)
print(test_data)
train_data.to_csv('source_list/final_list/'+src_class+'_train.csv')
test_data.to_csv('source_list/final_list/'+src_class+'_test.csv')