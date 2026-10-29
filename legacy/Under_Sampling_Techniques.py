from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss


import pandas as pd
from collections import Counter

def Random_Undersampling_tec(data):
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=0)
    X_resampled_RandUnder, y_resampled_RandUnder = rus.fit_resample(data.drop('Class', axis=1), data['Class'])
    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_RandUnder))
    data_resampled = pd.concat([X_resampled_RandUnder, y_resampled_RandUnder], axis=1)

    return data_resampled

def TomekLinks_tec(data):
    tl = TomekLinks(sampling_strategy='auto',)
    X_resampled_tomek, y_resampled_tomek = tl.fit_resample(data.drop('Class', axis=1), data['Class'])
    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_tomek))
    data_resampled = pd.concat([X_resampled_tomek, y_resampled_tomek], axis=1)

    return data_resampled

def ENN_tec(data):
    enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=31, kind_sel='all')
    X_resampled_enn, y_resampled_enn = enn.fit_resample(data.drop('Class', axis=1), data['Class'])
    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_enn))
    data_resampled = pd.concat([X_resampled_enn, y_resampled_enn], axis=1)

    return data_resampled

def NearMiss1_tec(data):
    nm1 = NearMiss(
            sampling_strategy='auto',  # undersamples only the majority class
            version=1,
            n_neighbors=3,
            n_jobs=4)
    X_resampled_NearMiss1, y_resampled_NearMiss1 = nm1.fit_resample(data.drop('Class', axis=1), data['Class'])
    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_NearMiss1))
    data_resampled = pd.concat([X_resampled_NearMiss1, y_resampled_NearMiss1], axis=1)

    return data_resampled

def NearMiss2_tec(data):
    nm1 = NearMiss(
            sampling_strategy='auto',  # undersamples only the majority class
            version=2,
            n_neighbors=3,
            n_jobs=4)
    X_resampled_NearMiss2, y_resampled_NearMiss2 = nm1.fit_resample(data.drop('Class', axis=1), data['Class'])
    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_NearMiss2))
    data_resampled = pd.concat([X_resampled_NearMiss2, y_resampled_NearMiss2], axis=1)

    return data_resampled
def NearMiss3_tec(data):
    nm1 = NearMiss(
            sampling_strategy='auto',  # undersamples only the majority class
            version=3,
            n_neighbors=3,
            n_jobs=4)
    X_resampled_NearMiss3, y_resampled_NearMiss3 = nm1.fit_resample(data.drop('Class', axis=1), data['Class'])
    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_NearMiss3))
    data_resampled = pd.concat([X_resampled_NearMiss3, y_resampled_NearMiss3], axis=1)

    return data_resampled





