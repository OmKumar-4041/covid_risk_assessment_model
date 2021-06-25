import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":

    # Read the data
    df = pd.read_csv('date.csv')
    train, test = data_split(df, 0.2)
    X_train = train[['gender', 'age_year', 'fever', 'cough', 'runny_nose','muscle_soreness','pneumonia','diarrhea','lung_infection','travel_history','isolation_treatment']].to_numpy()
    X_test = test[['gender', 'age_year', 'fever', 'cough', 'runny_nose','muscle_soreness','pneumonia','diarrhea','lung_infection','travel_history','isolation_treatment']].to_numpy()

    Y_train = train[['covid_positive']].to_numpy().reshape(5210 ,)
    Y_test = test[['covid_positive']].to_numpy().reshape(1302 ,)


    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
  

    file=open('model.pkl','wb')
    pickle.dump(clf, file)
    file.close()
    
  
   