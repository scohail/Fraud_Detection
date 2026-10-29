from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier




def grid_search(df,model):
    df_x = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)




    if(model=='svm'):
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        c_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]
        gammas = [0.1, 1, 10, 100]
        clf = SVC()
        clf.fit(X_train, y_train)
        param_grid = {'kernel': kernels, 'C': c_values, 'gamma': gammas}
        grid = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)

    elif(model=='DT'):
        param_grid = {
            'max_depth': [10, 20, 30, 40],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['auto', 'sqrt']
        }

        # Create a GridSearchCV object
        clf=DecisionTreeClassifier()
        grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, scoring='accuracy')



    elif(model=='knn'):
        neighbors = list(range(1, 21))
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': neighbors}
        grid = GridSearchCV(knn, param_grid, cv=10, n_jobs=-1)
        
    grid.fit(X_train, y_train)
    print("Best parameters for model",model,":", grid.best_params_)

    y_pred = grid.predict(X_test)
    
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    sns.heatmap((cnf_matrix / np.sum(cnf_matrix)*100),
            annot=True, fmt=".2f", cmap="Blues")

