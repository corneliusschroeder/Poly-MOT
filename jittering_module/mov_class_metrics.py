import pandas as pd

def check_matches(df):
    return ((df['Type'] == 'MATCH').sum() == len(df))

def accuracy(df):
    assert check_matches(df), 'df should only include match events'
    num_tp = (df['mov_Pred'] == df['mov_GT']).sum()
    num_preds = df['mov_Pred'].notnull().sum()
    acc = num_tp / num_preds
    return acc

def precision(df):
    assert check_matches(df), 'df should only include match events'
    num_tp = (df['mov_Pred'] == df['mov_GT']).sum()
    num_fp = ((df['mov_Pred'] == 'moving') & (df['mov_GT'] == 'static')).sum()
    prec = num_tp / (num_tp + num_fp)
    return prec

def recall(df):
    assert check_matches(df), 'df should only include match events'
    num_tp = (df['mov_Pred'] == df['mov_GT']).sum()
    num_fn = ((df['mov_Pred'] == 'static') & (df['mov_GT'] == 'moving')).sum()
    rec = num_tp / (num_tp + num_fn)
    return rec