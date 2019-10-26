import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

dfx = pd.read_csv('Diabetes_XTrain.csv')
dfy = pd.read_csv('Diabetes_YTrain.csv')
x_t=pd.read_csv('Diabetes_Xtest.csv')

x=dfx.values
y=dfy.values
x_t=x_t.values
y=y.reshape(576,)
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
# Test Time 
def knn(X,Y,queryPoint,k=5):
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred
len=x_t.shape[0] 
ls=[]
for i in range(len):   
    z=knn(x,y,x_t[i])
    z=list(str(int(z)))
    with open('test.csv','a') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerows(z)
    csvFile.close()    
    
