import numpy as np
import pandas as pd
import csv
from urllib import request
from io import StringIO
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn import datasets

##Load the data

url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes"
s = request.urlopen(url).read().decode('utf8')
dfile = StringIO(s)
creader = csv.reader(dfile)
dlists = [rw for rw in creader]
df = pd.DataFrame(dlists)
df = df[0].str.split(pat=None, n=-1, expand=True)
for i in range(9):
    temp = df[i].str.split(pat=':', n=-1, expand=True)
    if i != 0:
        df[i] = temp[1]
    else:
        df[i] = temp[0]
data = np.array(df).astype(np.float)
data = csr_matrix(data)
y = data.todense()[:, 0]
x = data.todense()[:, 1:]

## Partition to training and test sets

n = x.shape[0]
n_train = int(np.ceil(n * 0.8))
n_test = n - n_train
rand_indices = np.random.permutation(n)
train_indices = rand_indices[0:n_train]
test_indices = rand_indices[n_train:n]
x_train = x[train_indices, :]
x_test = x[test_indices, :]
y_train = y[train_indices].reshape(n_train, 1)
y_test = y[test_indices].reshape(n_test, 1)

# print('Shape of x_train: ' + str(x_train.shape))
# print('Shape of x_test: ' + str(x_test.shape))
# print('Shape of y_train: ' + str(y_train.shape))
# print('Shape of y_test: ' + str(y_test.shape))

## Features Scaling

d = x_train.shape[1]
mu = np.mean(x_train, axis=0).reshape(1, d)
sig = np.std(x_train, axis=0).reshape(1, d)
x_train = (x_train - mu) / (sig + 1E-6)
x_test = (x_test - mu) / (sig + 1E-6)

# print('test mean = ')
# print(np.mean(x_test, axis=0))
# print('test std = ')
# print(np.std(x_test, axis=0))

## make x bar
n_train, d = x_train.shape
x_train = np.concatenate((x_train, np.ones((n_train, 1))), axis=1)
n_test, d = x_test.shape
x_test = np.concatenate((x_test, np.ones((n_test, 1))), axis=1)

# print('Shape of x_train: ' + str(x_train.shape))
# print('Shape of x_test: ' + str(x_test.shape))

##Logistic regression model

def objective(w, x, y, lam):
    n, d = x.shape
    yx = np.multiply(y, x)
    yxw = np.dot(yx, w)
    vec1 = np.exp(-yxw)
    vec2 = np.log(1 + vec1)
    loss = np.mean(vec2)
    reg = (lam / 2) * np.sum(w * w)
    return loss + reg


##Initialize w
d = x_train.shape[1]
w = np.zeros((d, 1))

##Evaluate the objective function value at w
lam = 1E-6
objval0 = objective(w, x_train, y_train, lam)
print('Initial objective function value = ' + str(objval0))

##Gradient descent
def gradient(w, x, y, lam):
    n, d = x.shape
    yx = np.multiply(y, x)
    yxw = np.dot(yx, w)
    vec1 = np.exp(yxw)
    vec2 = np.divide(yx, 1 + vec1)
    vec3 = -np.mean(vec2, axis=0).reshape(d, 1)
    g = vec3 + lam * w
    return g


def grad_desccent(x, y, lam, stepsize, max_iter=100, w=None):
    n, d = x.shape
    objvals = np.zeros(max_iter)
    if w is None:
        w = np.zeros((d, 1))
    for t in range(max_iter):
        objval = objective(w, x, y, lam)
        objvals[t] = objval
        print('Objective value in GD at t= ' + str(t) + ' is ' + str(objval))
        g = gradient(w, x, y, lam)
        w -= stepsize * g
    return w, objvals

lam = 1E-6
stepsize = 1.0
w, objvals_gd = grad_desccent(x_train, y_train, lam, stepsize)
print('objvals_gd: '+str(objvals_gd[-1]))

## Stochastic gradient descent(SGD)

def stochastic_objective_gradient(w,xi,yi,lam):
    n,d=xi.shape
    yx=yi*xi
    yxw=float(np.dot(yx,w))
    loss=np.log(1+np.exp(-yxw))
    reg=(lam/2)*np.sum(w*w)
    obj=loss+reg
    g_loss=-yx.T/(1+np.exp(yxw))
    g=g_loss+lam*w
    return obj,g

def sgd(x,y,lam,stepsize,max_epoch=100,w=None):
    n,d=x.shape
    objvals=np.zeros(max_epoch)
    if w is None:
        w=np.zeros((d,1))
    for t in range(max_epoch):
        rand_indices=np.random.permutation(n)
        x_rand=x[rand_indices,:]
        y_rand=y[rand_indices,:]
        objval=0
        for i in range(n):
            xi=x_rand[i,:]
            yi=float(y_rand[i,:])
            obj,g=stochastic_objective_gradient(w,xi,yi,lam)
            objval+=obj
            w-=stepsize*g
        stepsize*=0.9
        objval /= n
        objvals[t]=objval
        print('Objective value in SGD at epoch t= '+str(t)+' is '+str(objval))
    return w,objvals

lam=1E-6
stepsize=0.1
w,objvals_sgd=sgd(x_train,y_train,lam,stepsize)
print('objvals_sgd: '+str(objvals_sgd[-1]))

##Prediction

def predict(w,x):
    xw=np.dot(x,w)
    f=np.sign(xw)
    return f

f_train=predict(w,x_train)
diff=np.abs(f_train-y_train)/2
error_train=np.mean(diff)
print('Training classification error in SGD is '+ str(error_train))

f_test=predict(w,x_test)
diff=np.abs(f_test-y_test)/2
error_test=np.mean(diff)
print('Test classification error is in SGD '+ str(error_test))


def mb_stochastic_objective_gradient(w,xi,yi,lam):
    d=xi.shape[1]
    yx=np.multiply(yi,xi)
    yxw=np.dot(yx,w)
    loss = np.mean(np.log(1 +np.exp(-yxw)),axis=0)
    reg = (lam / 2) * np.sum(w * w)
    obj = loss + reg
    vec1 = np.exp(yxw)
    vec2=np.divide(yx,1+vec1)
    vec3=-np.mean(vec2,axis=0).reshape(d,1)
    g=vec3+lam*w
    return obj,g

def mb_sgd(x,y,lam,b,stepsize,max_epoch=100,w=None):
    n,d=x.shape
    objvals=np.zeros(max_epoch)
    n=int(np.floor(n/b)*b)
    if w is None:
        w=np.zeros((d,1))
    for t in range(max_epoch):
        rand_indices=np.random.permutation(n)
        x_rand=x[rand_indices,:]
        y_rand=y[rand_indices,:]
        objval=0
        batch_size=0
        for batch in range(int(n/b)):
            xi=x_rand[batch_size:batch_size+b,:]
            # print(y_rand[batch_size:batch_size+b,:])
            yi=y_rand[batch_size:batch_size+b,:]
            obj,g=mb_stochastic_objective_gradient(w,xi,yi,lam)
            objval+=obj
            w-=stepsize*g
            batch_size=batch_size+b
        stepsize *= 0.9
        objvals[t]=objval/(int(n/b))
        print('Objective value in mb_SGD at epoch t= '+str(t)+' is '+str(objval/(int(n/b))))
    return w,objvals

lam=1E-6
b=8
stepsize=0.1
w_mb,objvals_mbsgd8=mb_sgd(x_train,y_train,lam,b,stepsize)
print('objvals_mbsgd8:'+ str(objvals_mbsgd8[-1]))

f_train=predict(w,x_train)
diff=np.abs(f_train-y_train)/2
error_train=np.mean(diff)
print('Training classification error in mbSGD8 is '+ str(error_train))

f_test=predict(w,x_test)
diff=np.abs(f_test-y_test)/2
error_test=np.mean(diff)
print('Test classification error in mbSGD8 is '+ str(error_test))

lam=1E-6
b=64
stepsize=0.1
w_mb,objvals_mbsgd64=mb_sgd(x_train,y_train,lam,b,stepsize)
print('objvals_mbsgd64 :'+str(objvals_mbsgd64[-1]))

f_train=predict(w,x_train)
diff=np.abs(f_train-y_train)/2
error_train=np.mean(diff)
print('Training classification error in mbSGD64 is '+ str(error_train))

f_test=predict(w,x_test)
diff=np.abs(f_test-y_test)/2
error_test=np.mean(diff)
print('Test classification error in mbSGD64 is '+ str(error_test))


fig=plt.figure(figsize=(6,4))
epochs_gd=range(len(objvals_gd))
epochs_sgd=range(len(objvals_sgd))
epochs_mbsgd8=range(len(objvals_mbsgd8))
epochs_mbsgd64=range(len(objvals_mbsgd64))
line0, =plt.plot(epochs_gd,objvals_gd,'--b',LineWidth=4)
line1, =plt.plot(epochs_sgd,objvals_sgd,'--r',LineWidth=2)
line2, =plt.plot(epochs_mbsgd8,objvals_mbsgd8,'--c',LineWidth=3)
line3, =plt.plot(epochs_mbsgd64,objvals_mbsgd64,'--y',LineWidth=1)
plt.xlabel('Epochs', FontSize=20)
plt.xticks(FontSize=16)
plt.yticks(FontSize=16)
plt.legend([line0,line1,line2,line3],['GD','SGD','mbSGB8','mbSGB64'],fontsize=20)
plt.tight_layout()
plt.show()