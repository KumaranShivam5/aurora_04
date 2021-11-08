# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns

# %% [markdown]
# ## Filter data

# %%
def filter_data(data_sent , max_flux= -12):
    data = data_sent.copy()
    #max_flux = -12
    min_flux = 26
    data = data[data['flux_aper']<max_flux]

    data = data[data['significance']>2]
    data_class = data[['class']]

    data_sig = data['significance']
    data_id = data['src_id']
    data_name = data['src_n']
    obs_info_params = [ 'livetime','likelihood','pileup_flag','mstr_sat_src_flag','mstr_streak_src_flag'   ,'gti_obs' , 'flux_significance_b'  , 'flux_significance_m' , 'flux_significance_s' , 'flux_significance_h' , 'flux_significance_u'    ]
    data_val = data.drop(columns=obs_info_params).reset_index(drop=True)
    return data_val

def norm_data(data_sent):
    data = data_sent.copy()
    #data.replace()
    for d in data:
        max_val = np.amax(data[d])
        min_val =  np.amin(data[d])
        data[d] = (data[d]-min_val)/(max_val-min_val)
    return data
def std_data(data_sent):
    data = data_sent.copy()
    for d in data:
        mean =  np.mean(data[d])
        std = np.sqrt(np.var(data[d]))
        data[d] = (data[d]-mean)/std 
    return data
def do_nothing(data_sent):
    return data_sent


# %%
def extract_data(data_sent , impute_fn = '',reduce_fn = ' ' , rf_impute=False):
    data = data_sent.copy()
    data = data.sample(frac=1)
    #data = filter_data(data)
    #print(data)
    data_id = data[['obs_id' ,'class' ,'src_n' , 'src_id' ,'significance' , ]]
    data_id = data_id.reset_index(drop=True)
    data_val = data.drop(['index' , 'class' ,'src_n' , 'src_id' ,'significance' , 'obs_id'] , axis=1)
    data_val = reduce_fn(data_val)
    return data_val , data_id
    #if(rf_impute):
    #    data_val  , random_forest_imputer = impute_fn(data_val , data_id)
    #else:
    #    data_val = impute_fn(data_val)
    data_val = reduce_fn(data_val)
    data_val = data_val.reset_index(drop=True)
    data_reduced = pd.concat([data_id , data_val] , axis=1)
    if(rf_impute):
        return(data_reduced , random_forest_imputer)
    else:
        return data_reduced

# %% [markdown]
# ## Load data

# %%

train_bh = pd.read_csv('../processed_data/BH_.csv')
train_bh = train_bh.sample(frac=1)
train_bh = filter_data(train_bh , max_flux=-12)


train_ns = pd.read_csv('../processed_data/NS_.csv' )
train_ns = train_ns.sample(frac=1)
train_ns = filter_data(train_ns , max_flux=-12)


# %%
train_cv = pd.read_csv('../processed_data/CV_.csv')
train_cv = train_cv.sample(frac=1)
train_cv = filter_data(train_cv , max_flux=-12)
sns.displot(train_cv['flux_aper'] , kde=True)


# %%

train_plsr = pd.read_csv('../processed_data/PULSAR_.csv')
train_plsr = train_plsr.sample(frac=1)
train_plsr = filter_data(train_plsr , max_flux=-10)
sns.displot(train_plsr['flux_aper'])
plt.show()


# %%
train_plsr['src_id'].value_counts()


# %%

train = pd.concat([train_bh , train_ns  , train_cv , train_plsr] , axis=0)
train = train.replace('NS' , 'XRB')
train = train.replace('BH' , 'XRB')
train =  train.sample(frac=1).reset_index(drop=True)
print(train)


# %%
train['class'].value_counts()

# %% [markdown]
# # RF imputer

# %%
from MissingValuesHandler.missing_data_handler import RandomForestImputer


# %%
def rf_impute(d, i ):
    data = pd.concat([i , d] , axis=1)
    data = data.drop(columns=['src_n' , 'src_id' , 'significance' , 'obs_id'])
    rf_imputer = RandomForestImputer(
        data=data , 
        target_variable_name='class' , 
        forbidden_features_list=[] , 
        
    )
    rf_imputer.set_ensemble_model_parameters(n_estimators=400 , additional_estimators = 100 )
    new_data =  rf_imputer.train(sample_size = 0, path_to_save_dataset='processed_data/rf_imp.csv')
    new_data = new_data.drop(columns= ['class'])
    return new_data , rf_imputer


# %%
data_val , data_id   = extract_data(train ,  impute_fn= rf_impute , reduce_fn= do_nothing , rf_impute=True )


# %%
new_data , random_forest_imputer = rf_impute(data_val, data_id)
#train_data.index.name = 'index'
#print(train_data.describe())
#train_data.to_csv('../processed_data/train_norm_rf_impute')


# %%
train


# %%
pd.read_csv('processed_data/rf_imp.csv')


# %%
data_id


# %%
sample_used                         = random_forest_imputer.get_sample()
features_type_prediction            = random_forest_imputer.get_features_type_predictions()
target_variable_type_prediction     = random_forest_imputer.get_target_variable_type_prediction()
encoded_features                    = random_forest_imputer.get_encoded_features()
encoded_target_variable             = random_forest_imputer.get_target_variable_encoded()
final_proximity_matrix              = random_forest_imputer.get_proximity_matrix()
final_distance_matrix               = random_forest_imputer.get_distance_matrix()
weighted_averages                   = random_forest_imputer.get_nan_features_predictions(option="all")
convergent_values                   = random_forest_imputer.get_nan_features_predictions(option="conv")
divergent_values                    = random_forest_imputer.get_nan_features_predictions(option="div")
ensemble_model_parameters           = random_forest_imputer.get_ensemble_model_parameters()
all_target_value_predictions        = random_forest_imputer.get_nan_target_values_predictions(option="all")
target_value_predictions            = random_forest_imputer.get_nan_target_values_predictions(option="one")


# %%
encoded_features_norm = norm_data(encoded_features)
encoded_features_norm


# %%
processed_data_all = pd.concat([data_id , pd.read_csv('processed_data/rf_imp.csv').drop(columns=['class'])] , axis=1)
processed_data_all.to_csv('../processed_data/train_norm_rf_impute_cv_xrb_pulsar')


# %%
processed_data_all = pd.concat([data_id , encoded_features] , axis=1)
processed_data_all.to_csv('../processed_data/train_none_rf_impute_cv_xrb_pulsar')


# %%
processed_data_all


# %%
plt.figure(figsize=(12,10))
plt.imshow(
    final_proximity_matrix , 
    cmap='gray'
)
plt.show()


