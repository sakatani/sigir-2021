import matplotlib.pyplot as plt
import numpy as np


def plot_loghist(x, bins):
    _, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')

def lag_feature(df, group_col, target_col, offset, nan=-1):
    sr = df[target_col].shift(offset)
    sr[df[group_col] != df[group_col].shift(offset)] = nan
    return sr.fillna(nan)

def log_feature(df, col):
    sr =  np.log1p(df[col])
    sr = sr - sr.mean()
    print(sr.mean(), sr.std())
    return sr

def count_in_session(df, count_col):
    tmp_df = df.groupby(count_col)['SessionId'].count().reset_index()
    tmp_df.columns = [count_col, count_col + '_count']
    df = df.merge(tmp_df, how='left', on=count_col)
    del tmp_df
    return df

def first_in_session(df, count_col, new_col_name, session_id='SessionId'):
    tmp_df = df[df['cum_product']==1][[session_id, count_col]]
    tmp_df.columns = [session_id, new_col_name]
    df = df.merge(tmp_df, on=session_id, how='left')
    del tmp_df
    return df