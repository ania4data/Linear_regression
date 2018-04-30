
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D   #3d plotting
import matplotlib.pyplot as plt


# In[269]:


X=[]
Y=[]

for line in open('data_2d.csv'):

    x1,x2,y=line.split(',')

    X.append([float(x1),float(x2),1])
    Y.append(float(y))

X=np.array(X)
Y=np.array(Y)


# In[270]:


plt.scatter(X[:,0],Y)
plt.scatter(X[:,1],Y)
plt.scatter(X[:,2],Y)


# In[271]:


#or

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)


# In[272]:


#(XT.X)w=XT.Y
w=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))


# In[273]:


#w


# In[274]:


y_new=np.dot(X,(w.T))


# In[276]:


#np.shape(y_new)


# In[277]:


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
ax.scatter(X[:,0],X[:,1],y_new)


# In[278]:


R_squared=1-(((Y-y_new).dot(Y-y_new))/((Y-Y.mean()).dot(Y-Y.mean())))


# In[279]:


#R_squared


# In[280]:


#np.shape(X)[1]


# In[281]:


#sigma * np.random.randn(...) + mu   For random samples from N(\mu, \sigma^2), use: 1/D variance
dim=np.shape(X)[1]
sigma_=np.sqrt(1/dim)
w_alpha=sigma_*(np.random.rand(dim))+0.0
#sigma_


# In[282]:


#w_alpha


# In[283]:


#X


# In[284]:


dim=np.shape(X)[0]
#dim


# In[285]:


X_norm=np.zeros((dim,np.shape(X)[1]))
Y_norm=np.zeros((np.shape(X)[0]))
np.max(Y)


# In[286]:



for i in range(0,np.shape(X)[1]-1):
    #print(np.mean(X[:,i]),np.max(X[:,i]),np.min(X[:,i]))
    X_norm[:,i]=(X[:,i]-np.mean(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
    #X_norm[:,i]=X[:,i]
X_norm[:,np.shape(X)[1]-1]=1.0

for i in range(0,np.shape(X)[0]-1):
    #Y_norm[i]=(Y[i]-np.mean(Y))/(np.max(Y)-np.min(Y))
     Y_norm[i]=Y[i]    #by normalizing X alone can converge, without X normalizing does not converge
#X_norm=X
#Y_norm=Y

# In[287]:


Y_norm=np.array(Y_norm)
X_norm=np.array(X_norm)
#print(X)
#Y_norm


# In[304]:


alpha_=0.001      #[[0.001,0.001,0.001]]

costs=[]
itr=[]
dim=np.shape(X)[1]
sigma_=np.sqrt(1/dim)
w_alpha=sigma_*(np.random.rand(3))+0.0
#print('0 iterate')
#print(w_alpha)
#print('iterate')
#print(X_norm)
for i in range(1000):

    y_new=X_norm.dot(w_alpha)
    delta=(y_new-Y_norm)
    w_alpha=w_alpha-alpha_*(X_norm.T.dot(delta))
    mse=(delta.dot(delta))/np.shape(X)[0]
    costs.append(mse)
    itr.append(i)
    print(w_alpha)
#    print(costs)

#print(np.shape(w_alpha))
#print(np.shape(X_norm.T))
#print(np.shape(X_norm))
#print(np.shape(y_new))
#print(delta)
#costs[9999]


# In[ ]:
#costs=np.array(costs)
#print(costs)
fig=plt.figure()
plt.plot(itr,costs)
plt.show()
#plt.scatter(X[:,1],Y)
#plt.scatter(X[:,2],Y)
#0.000672030933712375
#0.0036045686125814012
#0.0006720300664445527 10K
#plt.plot(costs[:,0],costs[:,1])
#plt.show()


# In[262]:


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X_norm[:,0],X_norm[:,1],Y_norm)
ax.scatter(X_norm[:,0],X_norm[:,1],y_new)
plt.show()

# In[265]:


R_squared=1-(((Y_norm-y_new).dot(Y_norm-y_new))/((Y_norm-Y_norm.mean()).dot(Y_norm-Y_norm.mean())))


# In[266]:


#R_squared
