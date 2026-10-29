""" Define the over-sampling methods. """



import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE


#problem with SMOTEENN
def SMOTEENN_tec(data):

    smoteenn = SMOTEENN(random_state=42)

    X_resampled_SMOTEENN, y_resampled_SMOTEENN = smoteenn.fit_resample(data.drop('Class', axis=1), data['Class'])

    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_SMOTEENN))  

    data_resampled = pd.concat([X_resampled_SMOTEENN, y_resampled_SMOTEENN], axis=1)

    return data_resampled

def SMOTE_tec(data): 
    smote= SMOTE(random_state=42)

    X_resampled_SMOTE, y_resampled_SMOTE = smote.fit_resample(data.drop('Class', axis=1), data['Class'])

    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_SMOTE))

    data_resampled = pd.concat([X_resampled_SMOTE, y_resampled_SMOTE], axis=1)
    return data_resampled

def RandomOverSampler_tec(data):
    ros = RandomOverSampler(random_state=42)

    X_resampled_RandomOverSampler, y_resampled_RandomOverSampler = ros.fit_resample(data.drop('Class', axis=1), data['Class'])

    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_RandomOverSampler))



    data_resampled = pd.concat([X_resampled_RandomOverSampler, y_resampled_RandomOverSampler], axis=1)

    return data_resampled

def ADASYN_tec(data):
    ada = ADASYN(random_state=42)

    X_resampled_ADASYN, y_resampled_ADASYN = ada.fit_resample(data.drop('Class', axis=1), data['Class'])

    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_ADASYN))
    data_resampled = pd.concat([X_resampled_ADASYN, y_resampled_ADASYN], axis=1)

    return data_resampled

def BorderlineSMOTE_tec(data):
    bsmote = BorderlineSMOTE(random_state=42)

    X_resampled_BorderlineSMOTE, y_resampled_BorderlineSMOTE = bsmote.fit_resample(data.drop('Class', axis=1), data['Class'])

    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_BorderlineSMOTE))
    
    
    data_resampled = pd.concat([X_resampled_BorderlineSMOTE, y_resampled_BorderlineSMOTE], axis=1)


    return data_resampled

def SVMSMOTE(data):
    svmsmote = SVMSMOTE(random_state=42)

    X_resampled_SVMSMOTE, y_resampled_SVMSMOTE = svmsmote.fit_resample(data.drop('Class', axis=1), data['Class'])

    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_SVMSMOTE))

    data_resampled = pd.concat([X_resampled_SVMSMOTE, y_resampled_SVMSMOTE], axis=1)

    return data_resampled

def KMeansSMOTE(data):
    kmeanssmote = KMeansSMOTE(random_state=42)

    X_resampled_KMeansSMOTE, y_resampled_KMeansSMOTE = kmeanssmote.fit_resample(data.drop('Class', axis=1), data['Class'])

    print('Original dataset shape %s' % Counter(data['Class']))
    print('Resampled dataset shape %s' % Counter(y_resampled_KMeansSMOTE))

    data_resampled = pd.concat([X_resampled_KMeansSMOTE, y_resampled_KMeansSMOTE], axis=1)


    return data_resampled




