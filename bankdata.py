import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
import pickle
# Load dataset
data= pd.read_csv("bank-full.csv",sep =";")
#data preprocessing
data.drop(columns =["day","month"],axis =1,inplace =True)
data.drop(columns="job",axis=1,inplace=True)
#labelling and encoding
le_marital = LabelEncoder()
le_education=LabelEncoder()
le_contact=LabelEncoder()
le_poutcome=LabelEncoder()
data['marital'] = le_marital.fit_transform(data['marital'])
data["education"]=le_education.fit_transform(data["education"])
data["contact"]=le_contact.fit_transform(data["contact"])
data["poutcome"]=le_poutcome.fit_transform(data["poutcome"])
data['housing'] = pd.get_dummies(data['housing'],dtype=int,drop_first=True)
data["loan"]=pd.get_dummies(data["loan"],dtype=int,drop_first=True)
data['default'] = pd.get_dummies(data['default'],dtype=int,drop_first=True)
data["y"]=pd.get_dummies(data["y"],dtype=int,drop_first=True)
with open('marital.pkl', 'wb') as f:
    pickle.dump(le_marital, f)

with open('education.pkl', 'wb') as f:
    pickle.dump(le_education, f)    

with open('contact.pkl', 'wb') as f:
    pickle.dump(le_contact, f)

with open('poutcome.pkl', 'wb') as f:
    pickle.dump(le_poutcome, f)

#splitting data
x = data.drop("y", axis=1)
y = data["y"]
#scaling
normalisation = MinMaxScaler()
x_scaled_array=normalisation.fit_transform(x)
# Coverting to Dataframe
x=pd.DataFrame(x_scaled_array,columns=x.columns)
with open('scaling.pkl', 'wb') as f:
    pickle.dump(normalisation, f)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =42,test_size=0.33)
K=3
selector = SelectKBest(score_func=f_classif, k=K)
x_train_selected = selector.fit_transform(x_train, y_train)
x_test_selected = selector.transform(x_test)
with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
selected_features = x_train.columns[selector.get_support()].tolist()
#handling imbalanced data
sm = SMOTE(random_state=42)
x_train_res, y_train_res = sm.fit_resample(x_train_selected, y_train)
with open('smote.pkl', 'wb') as f:
    pickle.dump(sm, f)
#model training
model = RandomForestClassifier(random_state=42)
model.fit(x_train_res, y_train_res)
#saving model
with open('RF_model.pkl', 'wb') as f:
    pickle.dump(model, f)   

