""" Define fuctions to apply the classification techniques. """

import pandas as pd
from sklearn.model_selection import cross_validate


# K_Fold Cross Validation

def K_fold_cross_validation_tec(data, model, k=10):
   
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    scoring = ['accuracy', 'precision', 'recall', 'f1']   
    scores = cross_validate(model , X, y , cv=k, scoring=scoring)

    # Convert the scores into a DataFrame for better readability
    scores_df = pd.DataFrame(scores)

    # Add a row for the mean of each score
    scores_df.loc['Mean'] = scores_df.mean()

    # Print the scores
    print(scores_df)

    return scores_df

#Decosion Tree Model

def DT_modele(data):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    return clf

#SVM Model

def SVM_modele(data):
    from sklearn.svm import SVC
    clf = SVC(random_state=42)
    return clf

#KNN Model

def KNN_modele(data):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    return clf

#Teste modele
def Test_model(data, model , balancing_tec):
    if balancing_tec == 'SMOTE':
        from Over_Sampling_Tec import SMOTE_tec
        data_resampled = SMOTE_tec(data)
    elif balancing_tec == 'RandomOverSampler': 
        from Over_Sampling_Tec import RandomOverSampler_tec   
        data_resampled = RandomOverSampler_tec(data)    
    elif balancing_tec == 'ADASYN':
        from Over_Sampling_Tec import ADASYN_tec
        data_resampled = ADASYN_tec(data)
    elif balancing_tec == 'BorderlineSMOTE':
        from Over_Sampling_Tec import BorderlineSMOTE_tec
        data_resampled = BorderlineSMOTE_tec(data)
    elif balancing_tec=='Undersampling':
        from Under_Sampling_Techniques import Random_Undersampling_tec
        data_resampled=Random_Undersampling_tec(data)
    elif balancing_tec=='Tomeklinks':
        from Under_Sampling_Techniques import Tomek_Links_tec
        data_resampled=Tomek_Links_tec(data)
    elif balancing_tec=='ENN':
        from Under_Sampling_Techniques import ENN_tec
        data_resampled=ENN_tec(data)
    elif balancing_tec=='Nearmiss1':
        from Under_Sampling_Techniques import NearMiss1_tec
        data_resampled=NearMiss1_tec(data)
    elif balancing_tec=='Nearmiss2':
        from Under_Sampling_Techniques import NearMiss2_tec
        data_resampled=NearMiss2_tec(data)
    elif balancing_tec=='Nearmiss3':
        from Under_Sampling_Techniques import NearMiss3_tec
        data_resampled=NearMiss3_tec(data)

    else:
        print('Error: The technique is not available')

    scores = K_fold_cross_validation_tec(data_resampled, model)

    
    return scores


