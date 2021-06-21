import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans,Birch,DBSCAN,AffinityPropagation,OPTICS
import matplotlib.pyplot as plt
import numpy as np
data2=pd.read_csv('Iris.csv')
data2
figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data2['SepalLengthCm'])
axr[1].boxplot(data2['SepalWidthCm'])
axr[0].set_title('SepalLengthCm')
axr[1].set_title('SepalWidthCm')
plt.show()
q1cl=data2['SepalWidthCm'].quantile(0.25)
q3cl=data2['SepalWidthCm'].quantile(0.75)
iqrcl=q3cl-q1cl
mincl=q1cl-(1*iqrcl)
maxcl=q3cl+(1*iqrcl)
print(mincl,maxcl)
figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data2['SepalLengthCm'])
axr[1].boxplot(data2['SepalWidthCm'])
axr[0].set_title('SepalLengthCm')
axr[1].set_title('SepalWidthCm')
plt.show()
figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data2['PetalLengthCm'])
axr[1].boxplot(data2['PetalWidthCm'])
axr[0].set_title('PetalLengthCm')
axr[1].set_title('PetalWidthCm')
plt.show()
data2['Species'].unique() #all unique cluster names
xcl=data2.iloc[:,1:-1].values
ycl=data2.iloc[:,-1:].values
xcl
clus1=KMeans(n_clusters=3,random_state=1).fit(xcl)
clus1
data2['clus']=clus1.predict(xcl)
data2['color']=data2.clus.map({0:'red',1:'yellow',2:'blue'})
data2
figcl=plt.figure(figsize=(21,15))
axcl=plt.axes(projection='3d')
axcl.scatter(data2['SepalLengthCm'],data2['SepalWidthCm'],data2['PetalLengthCm'],c=data2.color)
axcl.set_xlabel('SepalLengthCm')
axcl.set_ylabel('SepalWidthCm')
axcl.set_zlabel('PetalLengthCm')
plt.show()
figcl=plt.figure(figsize=(21,15))
axcl=plt.axes(projection='3d')
axcl.scatter(data2['SepalLengthCm'],data2['SepalWidthCm'],data2['PetalWidthCm'],c=data2.color)
axcl.set_xlabel('SepalLengthCm')
axcl.set_ylabel('SepalWidthCm')
axcl.set_zlabel('PetalWidthCm')
plt.show()
figcl=plt.figure(figsize=(21,15))
axcl=plt.axes(projection='3d')
axcl.scatter(data2['PetalWidthCm'],data2['SepalWidthCm'],data2['PetalLengthCm'],c=data2.color)
axcl.set_xlabel('PetalWidthCm')
axcl.set_ylabel('SepalWidthCm')
axcl.set_zlabel('PetalLengthCm')
plt.show()