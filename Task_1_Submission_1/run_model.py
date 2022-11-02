import os
import re
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import pickle

import shap
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from skbio.stats.composition import clr

import warnings
warnings.filterwarnings("ignore")

DIR_TRAIN = "data_train"
DIR_TEST = "input"

PATH_FTS_PHYLO_05 = "filtered_fts/phylotype_05_features.txt"
PATH_FTS_PHYLO_1 = "filtered_fts/phylotype_1_features.txt"
PATH_FTS_FAMILY = "filtered_fts/family_features.txt"
PATH_FTS_GENUS = "filtered_fts/genus_features.txt"
PATH_FTS_SPECIES = "filtered_fts/species_features.txt"

def load_fts_names(path):
    """
    Load feature names from txt files and return a list of strings.
    """
    def find_between(s, first, last ):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""

    res = []
    if "phylo" in path:
        with open(path,"r") as file:
            for line in file.readlines():
                res.extend([t[1:-1] for t in line[line.find("]")+1:].split()])
    else:
        with open(path,"r") as file:
            for line in file.readlines():
                res.append(find_between(line, '"', '"'))
    return res

def build_dictionaries(df_meta_tk):
    """
    Construct some mapping information
    """
    dic_pid_rids = {}
    dic_pid_specimen = {}
    dic_proj_pid = {}
    for rid, row in df_meta_tk.iterrows():
        if row['participant_id'] not in dic_pid_rids:
            dic_pid_rids[row['participant_id']] = [rid]
        else:
            dic_pid_rids[row['participant_id']].append(rid)
        if row['participant_id'] not in dic_pid_specimen:
            dic_pid_specimen[row['participant_id']] = [row['specimen']]
        else:
            dic_pid_specimen[row['participant_id']].append(row['specimen'])
        if row['project'] not in dic_proj_pid:
            dic_proj_pid[row['project']] = [row['participant_id']]
        else:
            dic_proj_pid[row['project']].append(row['participant_id'])
    dic_proj_pid = {i: set(dic_proj_pid[i]) for i in dic_proj_pid}
    return dic_pid_rids, dic_pid_specimen, dic_proj_pid


def weights_by_wk(dic_pid_specimen, dic_pid_rids, df_meta_tk):
    """
    More recent collections have heavier weights.
    Specimens from one participant sum up to 1.
    """
    dic_pid_clwk = {}
    for pid in dic_pid_specimen:
        dic_pid_clwk[pid] = [df_meta_tk[df_meta_tk['specimen']==sid]['collect_wk'].values[0] for sid in dic_pid_specimen[pid]]
    dic_pid_wts = {pid:[wk/sum(dic_pid_clwk[pid]) for wk in dic_pid_clwk[pid]] for pid in dic_pid_clwk}
    ary_weights = np.zeros(df_meta_tk.shape[0])
    for pid in dic_pid_specimen:
        ary_weights[dic_pid_rids[pid]] = dic_pid_wts[pid]
    return ary_weights

def add_encoded_age(df_meta_tk):
    """
    Convert age to categorical data and add to dataframe
    """
    lst_age = []
    for _, row in df_meta_tk.iterrows():
        try:
            age = float(row.age)
            if age < 18:
                age_cat = 'Below_18'
            elif age >= 18 and age < 28:
                age_cat = '18_to_28'
            elif age >= 28 and age <= 38:
                age_cat = '29-38'
            elif age > 38:
                age_cat = 'Above_38'
            else:
                age_cat = 'Unknown'
        except:
            age_cat = row.age
        lst_age.append(age_cat)
    df_meta_tk['age'] = df_meta_tk['age'].replace(['18_to_28', '29-38', 'Below_18', 'Above_38'], 'Unknown')
    df_meta_tk['age_cat'] = lst_age
    return df_meta_tk

def lgb_autotune(X_train, y_train, ary_weights_train, dic_pid_rids_train):
    # use Optuna to tune hyperparameters
    kf = KFold(n_splits=5, shuffle=True, random_state=41)
    lst_auroc = []
    def objective_lgb(trial):
        param = {
            'verbosity': trial.suggest_categorical('verbosity', [-1]),
            'objective': trial.suggest_categorical('objective', ['binary']),
            'num_iterations': trial.suggest_categorical('num_iterations', [25,50,75,100]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.02,0.03,0.04,0.05]),
            'max_depth': trial.suggest_categorical('max_depth', [3,4,5,6]),
            'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [16,20,24,28,30]),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-4, 1e-1),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 1e-1),
            'verbose': trial.suggest_categorical('verbose', [-1])
        }
        for train_ks, tune_ks in kf.split(dic_pid_rids_train):
            # participant IDs for training and test data
            train_pid = np.array(list(dic_pid_rids_train.keys()))[train_ks]
            tune_pid = np.array(list(dic_pid_rids_train.keys()))[tune_ks]
            # row indices for training data
            train_rid = [pp for pid in train_pid for pp in dic_pid_rids_train[pid]]
            # row indices for tuning data
            tune_rid = []
            dic_pid_tu_rid = {} # which rows are collected from which participants
            pointer = 0
            for pid in tune_pid:
                tune_rid.extend(dic_pid_rids_train[pid])
                dic_pid_tu_rid[pid] = list(range(pointer, pointer+len(dic_pid_rids_train[pid])))
                pointer += len(dic_pid_rids_train[pid])
            # split dataset
            assert [i for i in train_rid if i in tune_rid]==[], "Row IDs in training set also occur in test set."
            X_tr, y_tr = X_train_total[train_rid], y_train_total[train_rid]
            X_tu, y_tu = X_train_total[tune_rid], y_train_total[tune_rid]
            # build dataset
            lgb_tr = lgb.Dataset(X_tr, label=y_tr,
                                 categorical_feature=[2,3,4,5], weight=ary_weights_train[train_rid])
            gbm = lgb.train(param, lgb_tr,
                            categorical_feature=[2,3,4,5], init_model=None)
            preds_all = gbm.predict(X_tu)
            y_pred = [(ary_weights_train[tune_rid]*preds_all)[dic_pid_tu_rid[pid]].sum() for pid in dic_pid_tu_rid]
            y_true = [y_tu[dic_pid_tu_rid[pid]][0] for pid in dic_pid_tu_rid]
            lst_auroc.append((roc_auc_score(y_true, y_pred)))
        return np.mean(lst_auroc)
    sampler = TPESampler(seed=41)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective_lgb, n_trials=200)
    return study.best_params, study.best_value

dic_fts = {}
dic_fts["phylotype_05"] = load_fts_names(PATH_FTS_PHYLO_05)
dic_fts["phylotype_1"] = load_fts_names(PATH_FTS_PHYLO_1)
dic_fts["species"] = load_fts_names(PATH_FTS_SPECIES)
dic_fts["genus"] = load_fts_names(PATH_FTS_GENUS)
dic_fts["family"] = load_fts_names(PATH_FTS_FAMILY)

# read datasets
df_meta_train = pd.read_csv(f"{DIR_TRAIN}/metadata.csv")
df_alpha_diversity_train = pd.read_csv(f"{DIR_TRAIN}/alpha_diversity.csv")
df_community_state_train = pd.read_csv(f"{DIR_TRAIN}/cst_valencia.csv")
df_phylo05_train = pd.read_csv(f"{DIR_TRAIN}/phylotype_nreads.5e_1.csv")
df_phylo05_train[df_phylo05_train.columns[1:]] = clr(df_phylo05_train[df_phylo05_train.columns[1:]].values+0.5)
df_phylo1_train = pd.read_csv(f"{DIR_TRAIN}/phylotype_nreads.1e0.csv")
df_phylo1_train[df_phylo1_train.columns[1:]] = clr(df_phylo1_train[df_phylo1_train.columns[1:]].values+0.5)
df_species_train = pd.read_csv(f"{DIR_TRAIN}/taxonomy_nreads.species.csv")
df_species_train[df_species_train.columns[1:]] = clr(df_species_train[df_species_train.columns[1:]].values+0.5)
df_genus_train = pd.read_csv(f"{DIR_TRAIN}/taxonomy_nreads.genus.csv")
df_genus_train[df_genus_train.columns[1:]] = clr(df_genus_train[df_genus_train.columns[1:]].values+0.5)
df_family_train = pd.read_csv(f"{DIR_TRAIN}/taxonomy_nreads.family.csv")
df_family_train[df_family_train.columns[1:]] = clr(df_family_train[df_family_train.columns[1:]].values+0.5)

# task 1 only use data collected no later than 32 weeks
df_meta_tk1_train = df_meta_train[(df_meta_train["collect_wk"]<=32)].reset_index(drop=True)

# construct some dictionaries
dic_pid_rids_train, dic_pid_specimen_train, dic_proj_pid = build_dictionaries(df_meta_tk1_train)

# weights by date
ary_weights_train = weights_by_wk(dic_pid_specimen_train, dic_pid_rids_train, df_meta_tk1_train)

# add categorical age
df_meta_tk1_train = add_encoded_age(df_meta_tk1_train)

# encode categorical data
cols_demo = ['participant_id', 'specimen', 'project', 'collect_wk', 'NIH Racial Category', 'age', 'age_cat', 'was_term']
df_demo_train = df_meta_tk1_train[cols_demo]#.loc[df_meta_tk1.groupby("participant_id")["collect_wk"].idxmax()][cols_demo]
df_demo_train = df_demo_train.replace('Unknown', np.nan)
df_demo_train = df_demo_train.reset_index(drop=True)
df_demo_train['age'] = df_demo_train['age'].astype(float)
cat_cols = ['NIH Racial Category','age_cat','CST','subCST']
df_cat_train = df_demo_train[['specimen','NIH Racial Category','age_cat']].join(df_community_state_train[['specimen','CST','subCST']].set_index('specimen'), on='specimen')
# load pre-fit label encoder
with open("dic_encoder.pickle", "rb") as handle:
    dic_encoder = pickle.load(handle)
for col in cat_cols:
    le = dic_encoder[col]
    df_cat_train[col] = [x if x in le.classes_ else 'nan' for x in df_cat_train[col]]
    df_cat_train[col] = le.transform(df_cat_train[[col]])

# join datasets on specimen IDs
df_alpha_diversity_train = df_demo_train[['specimen']].join(df_alpha_diversity_train.set_index('specimen'), on='specimen')
df_phylo05_train = df_demo_train[['specimen']].join(df_phylo05_train.set_index('specimen'), on='specimen')
df_phylo1_train = df_demo_train[['specimen']].join(df_phylo1_train.set_index('specimen'), on='specimen')
df_species_train = df_demo_train[['specimen']].join(df_species_train.set_index('specimen'), on='specimen')
df_genus_train = df_demo_train[['specimen']].join(df_genus_train.set_index('specimen'), on='specimen')
df_family_train = df_demo_train[['specimen']].join(df_family_train.set_index('specimen'), on='specimen')
print("Training data loaded!")

# read datasets
df_meta_test = pd.read_csv(f"{DIR_TEST}/metadata/metadata.csv")
df_alpha_diversity_test = pd.read_csv(f"{DIR_TEST}/alpha_diversity/alpha_diversity.csv")
df_community_state_test = pd.read_csv(f"{DIR_TEST}/community_state_types/cst_valencia.csv")
df_phylo05_test = pd.read_csv(f"{DIR_TEST}/phylotypes/phylotype_nreads.5e_1.csv")
df_phylo05_test[df_phylo05_test.columns[1:]] = clr(df_phylo05_test[df_phylo05_test.columns[1:]].values+0.5)
df_phylo1_test = pd.read_csv(f"{DIR_TEST}/phylotypes/phylotype_nreads.1e0.csv")
df_phylo1_test[df_phylo1_test.columns[1:]] = clr(df_phylo1_test[df_phylo1_test.columns[1:]].values+0.5)
df_species_test = pd.read_csv(f"{DIR_TEST}/taxonomy/taxonomy_nreads.species.csv")
df_species_test[df_species_test.columns[1:]] = clr(df_species_test[df_species_test.columns[1:]].values+0.5)
df_genus_test = pd.read_csv(f"{DIR_TEST}/taxonomy/taxonomy_nreads.genus.csv")
df_genus_test[df_genus_test.columns[1:]] = clr(df_genus_test[df_genus_test.columns[1:]].values+0.5)
df_family_test = pd.read_csv(f"{DIR_TEST}/taxonomy/taxonomy_nreads.family.csv")
df_family_test[df_family_test.columns[1:]] = clr(df_family_test[df_family_test.columns[1:]].values+0.5)
print("Test data loaded!")
# task 1 only use data collected no later than 32 weeks
df_meta_tk1_test = df_meta_test[(df_meta_test["collect_wk"]<=32)].reset_index(drop=True)

# construct some dictionaries
dic_pid_rids_test, dic_pid_specimen_test, dic_proj_pid_test = build_dictionaries(df_meta_tk1_test)

# weights by date
ary_weights_test = weights_by_wk(dic_pid_specimen_test, dic_pid_rids_test, df_meta_tk1_test)

# add categorical age
df_meta_tk1_test = add_encoded_age(df_meta_tk1_test)

# encode categorical data
cols_demo = ['participant_id', 'specimen', 'project', 'collect_wk', 'NIH Racial Category', 'age', 'age_cat']
df_demo_test = df_meta_tk1_test[cols_demo]#.loc[df_meta_tk1.groupby("participant_id")["collect_wk"].idxmax()][cols_demo]
df_demo_test = df_demo_test.replace('Unknown', np.nan)
df_demo_test = df_demo_test.reset_index(drop=True)
df_demo_test['age'] = df_demo_test['age'].astype(float)
cat_cols = ['NIH Racial Category','age_cat','CST','subCST']
df_cat_test = df_demo_test[['specimen','NIH Racial Category','age_cat']].join(df_community_state_test[['specimen','CST','subCST']].set_index('specimen'), on='specimen')
# label encoding and save encoder as a dictionary
for col in cat_cols:
    le = dic_encoder[col]
    df_cat_test[col] = [x if x in le.classes_ else 'nan' for x in df_cat_test[col]]
    df_cat_test[col] = le.transform(df_cat_test[[col]])

# join datasets on specimen IDs
df_alpha_diversity_test = df_demo_test[['specimen']].join(df_alpha_diversity_test.set_index('specimen'), on='specimen')
df_phylo05_test = df_demo_test[['specimen']].join(df_phylo05_test.set_index('specimen'), on='specimen')
df_phylo1_test = df_demo_test[['specimen']].join(df_phylo1_test.set_index('specimen'), on='specimen')
df_species_test = df_demo_test[['specimen']].join(df_species_test.set_index('specimen'), on='specimen')
df_genus_test = df_demo_test[['specimen']].join(df_genus_test.set_index('specimen'), on='specimen')
df_family_test = df_demo_test[['specimen']].join(df_family_test.set_index('specimen'), on='specimen')

# only use features that orrur in test set
dic_fts['phylotype_05'] = [ft for ft in dic_fts['phylotype_05'] if ft in df_phylo05_test.columns]
dic_fts['phylotype_1'] = [ft for ft in dic_fts['phylotype_1'] if ft in df_phylo1_test.columns]
dic_fts['species'] = [ft for ft in dic_fts['species'] if ft in df_species_test.columns]
dic_fts['genus'] = [ft for ft in dic_fts['genus'] if ft in df_genus_test.columns]
dic_fts['family'] = [ft for ft in dic_fts['family'] if ft in df_family_test.columns]
df_phylo05_train = df_phylo05_train[['specimen']+dic_fts['phylotype_05']]
df_phylo05_test = df_phylo05_test[['specimen']+dic_fts['phylotype_05']]
df_phylo1_train = df_phylo1_train[['specimen']+dic_fts['phylotype_1']]
df_phylo1_test = df_phylo1_test[['specimen']+dic_fts['phylotype_1']]
df_species_train = df_species_train[['specimen']+dic_fts['species']]
df_species_test = df_species_test[['specimen']+dic_fts['species']]
df_genus_train = df_genus_train[['specimen']+dic_fts['genus']]
df_genus_test = df_genus_test[['specimen']+dic_fts['genus']]
df_family_train = df_family_train[['specimen']+dic_fts['family']]
df_family_test = df_family_test[['specimen']+dic_fts['family']]

# merge features
X_train_total = np.concatenate([df_demo_train[['collect_wk', 'age']],
                                df_cat_train.values[:,1:], # 'NIH Racial Category','age_cat','CST','subCST'
                                df_alpha_diversity_train.values[:,1:],
                                df_phylo05_train.values[:,1:],
                                df_phylo1_train.values[:,1:],
                                df_species_train.values[:,1:],
                                df_genus_train.values[:,1:],
                                df_family_train.values[:,1:]], axis=1)
print(X_train_total.shape)
# labels
y_train_total = np.zeros(df_demo_train.shape[0])
for rid, row in df_demo_train.iterrows():
    if row['was_term'] == False:
        y_train_total[rid] = 1

# test
X_test = np.concatenate([df_demo_test[['collect_wk', 'age']],
                          df_cat_test.values[:,1:], # 'NIH Racial Category','age_cat','CST','subCST'
                          df_alpha_diversity_test.values[:,1:],
                          df_phylo05_test.values[:,1:],
                          df_phylo1_test.values[:,1:],
                          df_species_test.values[:,1:],
                          df_genus_test.values[:,1:],
                          df_family_test.values[:,1:]], axis=1)

# tuning
params, cv_score = lgb_autotune(X_train_total, y_train_total, ary_weights_train, dic_pid_rids_train)

# train model 1
lgb_train1 = lgb.Dataset(X_train_total, label=y_train_total,
                        categorical_feature=[2,3,4,5], weight=ary_weights_train)
gbm1 = lgb.train(params, lgb_train1,
                 categorical_feature=[2,3,4,5], init_model=None)
# predict
y_pred_all_test1 = gbm1.predict(X_test)

# data for model 2
rid_G = [rid for pid in dic_proj_pid['G'] for rid in dic_pid_rids_train[pid]]
X_train_G = X_train_total[rid_G]
y_train_G = y_train_total[rid_G]
ary_weights_G = ary_weights_train[rid_G]

dic_pid_G_rid = {}
pointer = 0
for pid in dic_proj_pid['G']:
    dic_pid_G_rid[pid] = list(range(pointer, pointer+len(dic_pid_rids_train[pid])))
    pointer += len(dic_pid_rids_train[pid])

# tuning
params_G, cv_score = lgb_autotune(X_train_G, y_train_G, ary_weights_G, dic_pid_G_rid)

# train model 2
lgb_train2 = lgb.Dataset(X_train_G, label=y_train_G,
                        categorical_feature=[2,3,4,5], weight=ary_weights_G)
gbm2 = lgb.train(params_G, lgb_train2,
                 categorical_feature=[2,3,4,5], init_model=None)
# predict
y_pred_all_test2 = gbm2.predict(X_test)

# train a classifier that predict the prob that a sample is from G
df_counts_train = pd.read_csv(f"{DIR_TRAIN}/phylotype_nreads.5e_1.csv")
df_counts_train = df_demo_train[['specimen']].join(df_counts_train.set_index('specimen'), on='specimen')
X_cls_G_train = np.concatenate([np.sum(df_counts_train.values[:,1:], axis=1).reshape(-1,1),
                          X_train_total[:,0].reshape(-1,1)], axis=1)
y_cls_G_train = np.zeros(X_cls_G_train.shape[0])
y_cls_G_train[[rid for pid in dic_pid_rids_train if "G" in pid for rid in dic_pid_rids_train[pid]]] = 1
X_cls_G_train, y_cls_G_train = shuffle(X_cls_G_train, y_cls_G_train, random_state=0)
model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5, multi_class='ovr',random_state=0)
model.fit(X_cls_G_train, y_cls_G_train)
df_counts_test = pd.read_csv(f"{DIR_TEST}/phylotypes/phylotype_nreads.5e_1.csv")
df_counts_test = df_demo_test[['specimen']].join(df_counts_test.set_index('specimen'), on='specimen')
X_cls_G_test = np.concatenate([np.sum(df_counts_test.values[:,1:], axis=1).reshape(-1,1),
                          X_test[:,0].reshape(-1,1)], axis=1)
y_cls_G_test = np.zeros(X_cls_G_test.shape[0])
y_cls_G_test[[rid for pid in dic_pid_rids_test if "G" in pid for rid in dic_pid_rids_test[pid]]] = 1
pred_G = model.predict_proba(X_cls_G_test)[:,1]

# ensemble using above weights
prob_isG = np.minimum(pred_G, 0.8)
pred_all_emsemble = y_pred_all_test1*(1-prob_isG)+y_pred_all_test2*prob_isG
y_pred_ensemle = [(ary_weights_test*pred_all_emsemble)[dic_pid_rids_test[pid]].sum() for pid in dic_pid_rids_test]

# make output files
pid_test = [pid for pid in dic_pid_rids_test]
y_binary = [1 if y >= 0.5 else 0 for y in y_pred_ensemle]
dic_csv = {
    "participant": pid_test,
    "was_preterm": y_binary,
    "probability": y_pred_ensemle
}
df_out = pd.DataFrame(dic_csv)
df_out.to_csv('output/predictions.csv', index=False)
print("Done!")
