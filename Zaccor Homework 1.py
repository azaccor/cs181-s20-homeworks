#!/usr/bin/env python
# coding: utf-8

# # CS 181, Spring 2020
# # Homework 1
# # Austin Zaccor

# ## Problem 1

# In[1]:


import csv
import math
from math import exp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as c
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


x1 = [0,0,0,.5,.5,.5,1,1,1]
x2 = [0,.5,1,0,.5,1,0,.5,1]
y = [0,0,0,.5,.5,.5,1,1,1]


# In[3]:


a = 10
W1 = a*np.array([[1,0],[0,1]])
W2 = a*np.array([[0.1, 0], [0,1]])
W3 = a*np.array([[1,0],[0,0.1]])
W4 = a*np.array([[10,0],[0,0]])


# In[4]:


def mahalanobis_loss(kernel):
    N = len(y)
    loss1 = 0
    loss2 = 0.0000000000001
    loss = 0
    for i in range(0,N):
        loss1 = 0
        loss2 = 0.0000000000001
        for n in range(0,N):
            if i == n:
                continue
            else:
                loss1+=exp(-(kernel[0][0]*(x1[i]-x1[n])**2+
                                  kernel[0][1]*(x2[i]-x2[n])*(x1[i]-x1[n])+
                                  kernel[1][1]*(x2[i]-x2[n])**2))*y[n]
                loss2+=exp(-(kernel[0][0]*(x1[i]-x1[n])**2+
                                  kernel[0][1]*(x2[i]-x2[n])*(x1[i]-x1[n])+
                                  kernel[1][1]*(x2[i]-x2[n])**2))
        loss+=(y[i] - loss1/loss2)**2
    return(loss/2)


# In[5]:


print(mahalanobis_loss(W1))
print(mahalanobis_loss(W2))
print(mahalanobis_loss(W3))
print(mahalanobis_loss(W4))


# Since $x1 = y$, the more we weight the first entry of the kernel relative to the last entry, the more weight we are giving the difference between x1 and y and the less weight we give to the difference between x2 and y, which are much less correlated. 
# 
# Similarly for alpha, if we expand alpha, we can see that W3 improves, while W1 is made worse. W2 gets better momentarily, but eventually is worse, which makes sense given the fact that the first entry of W2 is smaller than the last, we would expect growth in alpha to only exacerbate this kernel's errors. W3 should be forever better off by expanding alpha. My fictional W4 should essentially be perfect, and likely doesn't equal 0 due to rounding only.

# ## Problem 2

# In[6]:


# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values


# In[7]:


def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    W1 = alpha*np.array([[1,0],[0,1]])
    x1 = X_df["x1"].values
    x2 = X_df["x2"].values
    N = len(y)
    numer = 0
    denom = 0
    y_hat = []
    for i in range(0,N):
        for n in range(0,N):
            if i == n:
                continue
            else:
                numer+=exp(-(W1[0][0]*(x1[i]-x1[n])**2+
                         W1[0][1]*(x2[i]-x2[n])*(x1[i]-x1[n])+
                         W1[1][1]*(x2[i]-x2[n])**2))*y[n]
                denom+=exp(-(W1[0][0]*(x1[i]-x1[n])**2+
                         W1[0][1]*(x2[i]-x2[n])*(x1[i]-x1[n])+
                         W1[1][1]*(x2[i]-x2[n])**2))
        y_hat.append(numer/denom)
        numer = 0
        denom = 0
                
    return(y_hat)


# In[8]:


def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    ## Return the indices of the nearest neighbors to be used in the y_hat prediction
    if k > 12:
        k = 12
    else:
        pass
    
    W1 = np.array([[1,0],[0,1]])
    N = len(y)
    x1 = X_df["x1"].values
    x2 = X_df["x2"].values
    KNN_mat = []
    for i in range(0,N):
        largest = []
        index_l = []
        for n in range(0,N):
            if i == n:
                continue
            else:
                l = exp(-(W1[0][0]*(x1[i]-x1[n])**2+W1[0][1]*(x2[i]-x2[n])*(x1[i]-x1[n])+W1[1][1]*(x2[i]-x2[n])**2))
            
            if len(largest) < k:
                largest.append(l)
                index_l.append(n)
            elif l > min(largest) and len(largest) >= k:
                idx_remove = largest.index(min(largest))
                largest.remove(min(largest))
                largest.append(l)
                index_l.pop(idx_remove)
                index_l.append(n)
            else:
                pass
        KNN_mat.append(index_l)
    KNN_mat = np.reshape(KNN_mat, (N,k))

    ## Return the actual y_hat predictions
    numer = 0
    denom = 0
    y_hat = []
    for i in range(0,N):
        x2_tmp = []
        x1_tmp = []
        y_tmp = []
        indices = KNN_mat[i]
    
        ## This block is how you get only the k nearest x1 and x2 values
        for j in range(0,k):
            x1_tmp.append(x1[indices[j]])
            x2_tmp.append(x2[indices[j]])
            y_tmp.append(y[indices[j]])
    
        ## Now we want to return y hat based on the x1_tmp and x2_tmp
        numer+=exp(-(W1[0][0]*(x1[i]-x1_tmp[j])**2+
                 W1[0][1]*(x2[i]-x2_tmp[j])*(x1[i]-x1_tmp[j])+
                 W1[1][1]*(x2[i]-x2_tmp[j])**2))*y_tmp[j]
        denom+=exp(-(W1[0][0]*(x1[i]-x1_tmp[j])**2+
                 W1[0][1]*(x2[i]-x2_tmp[j])*(x1[i]-x1_tmp[j])+
                 W1[1][1]*(x2[i]-x2_tmp[j])**2))
        y_hat.append(numer/denom)
        numer = 0
        denom = 0
    return(y_hat)


# In[9]:


def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')

    # Saving the image to a file, and showing it as well
    plt.savefig(title + '.png')
    plt.show()


# In[10]:


def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')

    # Saving the image to a file, and showing it as well
    plt.savefig(title + '.png')
    plt.show()


# In[11]:


for alpha in (0.1, 3, 10):
    loss = sum((y-np.array(predict_kernel(alpha)))**2)
    print("Loss for alpha = ", alpha, " is ", loss)
    plot_kernel_preds(alpha)


# In[12]:


for k in (1, 5, 15):
    loss = sum((y-np.array(predict_knn(k)))**2)
    print("Loss for alpha = ", alpha, " is ", loss)
    plot_knn_preds(k)


# ### Part 2
# Alpha = 10 kernel regression looks fairly similar to k = 1 KNN. This is likely due to the fact that the alpha = 10 regression is putting a lot of weight on the first and last entries of the kernel matrix. Since X2 is not very correlated with y but X1 is very correlated with y, we expect lower loss where X1 is high but X2 is low, which is what these lighter points represent - we're predicting a higher y_hat which is close to the true y. In the case of KNN, it is merely the fact that the 1 KNN does the best, and therefore the points in the bottom right are the only ones that show a high y_hat, whereas for larger K values, we get all of them showing up as large y_hats.

# ### Part 3
# No, I don't think so. I don't think that kernel regression can always replicate the results of KNN, because KNN is not bounded by the constraint of only making linear adjustments with respect to one of the dimensions. KNN is a totally nonparametric formula and therefore can take on very odd separating boundaries between two classes or make more drastic changes faster than kernel regression can.

# ### Part 4
# For one, if we had varied alpha during the KNN approach we would not be doing a very good controlled test. We were trying to compare different alphas for kernel regression against different k values for KNN regression, and introducing a second variable in the KNN regression spoils the experiment somewhat. 
# 
# But more importantly, and actually in contrast to my first point, it wouldn't have mattered. Changing alpha changes the y_hat output of the KNN method very little. In fact, even increasing the alpha by an order of magnitude or more only changes the output by a rounding error. For KNN, the points are decided and then only these k points are utilized in the prediction. 

# ## Problem 3

# See LaTex

# ## Problem 4

# In[13]:


csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []


# In[14]:


with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))


# In[15]:


# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985


# In[16]:


# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# In[17]:


# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, w)

y_hat = []
for i in range(0,len(grid_years)):
    if i == 0:
        y_hat.append(grid_Yhat[i])
    elif i % 9 == 0:
        y_hat.append(grid_Yhat[i])
    elif i == len(grid_years)-1:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares is approximately:", sum((y_hat-republican_counts)**2))

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ## Part a)

# In[54]:


# Create the basis for a
X = np.vstack((np.ones(years.shape), years, years**2, years**3, years**4, years**5)).T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X =np.vstack((np.ones(grid_years.shape), grid_years, grid_years**2, grid_years**3, grid_years**4, grid_years**5))
grid_Yhat  = np.dot(grid_X.T, w)

y_hat = []
for i in range(0,len(grid_years)):
    if i == 0:
        y_hat.append(grid_Yhat[i])
    elif i % 9 == 0:
        y_hat.append(grid_Yhat[i])
    elif i == len(grid_years)-1:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares ~", sum((y_hat-republican_counts)**2))

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ## Part b)

# In[56]:


mu_j = [1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
    
terms = []
for j in range(0, 11):
    for i in range(0, 24):
        terms.append(math.exp(-(years[i] - mu_j[j])**2/25))
terms = np.reshape(terms, (11, 24))


# In[57]:


mu_j = [1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
    
grid_terms = []
for j in range(0, 11):
    for i in range(0, 200):
        grid_terms.append(math.exp(-(grid_years[i] - mu_j[j])**2/25))
grid_terms = np.reshape(grid_terms, (11, 200))


# In[58]:


# Create the basis for b
X = np.vstack((np.ones(years.shape), terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], terms[6], terms[7], 
               terms[8], terms[9], terms[10])).T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_terms[0], grid_terms[1], grid_terms[2], 
                    grid_terms[3], grid_terms[4], grid_terms[5], grid_terms[6], grid_terms[7], 
                    grid_terms[8], grid_terms[9], grid_terms[10]))

grid_Yhat  = np.dot(grid_X.T, w)

y_hat = []
for i in range(0,len(grid_years)):
    if i == 0:
        y_hat.append(grid_Yhat[i])
    elif i % 9 == 0:
        y_hat.append(grid_Yhat[i])
    elif i == len(grid_years)-1:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares ~", sum((y_hat-republican_counts)**2))

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ## Part c)

# In[59]:


j_vec = [1,2,3,4,5]
    
terms = []
for j in range(0, 5):
    for i in range(0, 24):
        terms.append(math.cos(years[i]/j_vec[j]))
terms = np.reshape(terms, (5, 24))


# In[60]:


j_vec = [1,2,3,4,5]
    
grid_terms = []
for j in range(0, 5):
    for i in range(0, 200):
        grid_terms.append(math.cos(grid_years[i]/j_vec[j]))
grid_terms = np.reshape(grid_terms, (5, 200))


# In[61]:


# Create the basis for b
X = np.vstack((np.ones(years.shape), terms[0], terms[1], terms[2], terms[3], terms[4])).T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_terms[0], grid_terms[1], grid_terms[2], 
                    grid_terms[3], grid_terms[4]))

grid_Yhat  = np.dot(grid_X.T, w)

y_hat = []
for i in range(0,len(grid_years)):
    if i == 0:
        y_hat.append(grid_Yhat[i])
    elif i % 9 == 0:
        y_hat.append(grid_Yhat[i])
    elif i == len(grid_years)-1:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares ~", sum((y_hat-republican_counts)**2))

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ## Part d)

# In[62]:


j_vec = list(np.arange(1,25+1,1))
    
terms = []
for j in range(0, 25):
    for i in range(0, 24):
        terms.append(math.cos(years[i]/j_vec[j]))
terms = np.reshape(terms, (25, 24))


# In[63]:


j_vec = list(np.arange(1,25+1,1))
    
grid_terms = []
for j in range(0, 25):
    for i in range(0, 200):
        grid_terms.append(math.cos(grid_years[i]/j_vec[j]))
grid_terms = np.reshape(grid_terms, (25, 200))


# In[64]:


# Create the basis for b
X = np.vstack((np.ones(years.shape), terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], terms[6], terms[7],
               terms[8], terms[9], terms[10], terms[11], terms[12], terms[13], terms[14], terms[15], terms[16],
               terms[17], terms[18], terms[19], terms[20], terms[21], terms[22], terms[23], terms[24])).T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_terms[0], grid_terms[1], grid_terms[2], grid_terms[3], 
                    grid_terms[4], grid_terms[5], grid_terms[6], grid_terms[7], grid_terms[8], grid_terms[9], 
                    grid_terms[10], grid_terms[11], grid_terms[12], grid_terms[13], grid_terms[14], grid_terms[15], 
                    grid_terms[16], grid_terms[17], grid_terms[18], grid_terms[19], grid_terms[20], grid_terms[21], 
                    grid_terms[22], grid_terms[23], grid_terms[24]))

grid_Yhat  = np.dot(grid_X.T, w)

y_hat = []
for i in range(0,len(grid_years)):
    if i == 0:
        y_hat.append(grid_Yhat[i])
    elif i % 9 == 0:
        y_hat.append(grid_Yhat[i])
    elif i == len(grid_years)-1:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares ~", sum((y_hat-republican_counts)**2))

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ## Part 2
# ### a)

# In[86]:


# Create the basis for a
X = np.vstack((np.ones(sunspot_counts[years<last_year].shape), sunspot_counts[years<last_year], 
               sunspot_counts[years<last_year]**2, sunspot_counts[years<last_year]**3, 
               sunspot_counts[years<last_year]**4, sunspot_counts[years<last_year]**5)).T

# Nothing fancy for outputs.
Y = republican_counts[years<last_year]

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_sun = np.linspace(0, 160, 111)
grid_X =np.vstack((np.ones(grid_sun.shape), grid_sun, grid_sun**2, grid_sun**3, grid_sun**4, grid_sun**5))
grid_Yhat  = np.dot(grid_X.T, w)

vec = [  0,   9,  18,  27,  36,  45,  54,  63,  72,  81,  90,  99, 108]
y_hat = []
for i in range(0,len(grid_sun)):
    if i in vec:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares ~", sum((y_hat-republican_counts[years<last_year])**2))

# Plot the data and the regression line.
plt.plot(sunspot_counts, republican_counts, 'o', grid_sun, grid_Yhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ### Part c)

# In[66]:


j_vec = [1,2,3,4,5]
    
terms = []
for j in range(0, 5):
    for i in range(0, 13):
        terms.append(math.cos(sunspot_counts[years<last_year][i]/j_vec[j]))
terms = np.reshape(terms, (5, 13))


# In[67]:


j_vec = [1,2,3,4,5]
    
grid_terms = []
for j in range(0, 5):
    for i in range(0, 111):
        grid_terms.append(math.cos(grid_sun[grid_years<last_year][i]/j_vec[j]))
grid_terms = np.reshape(grid_terms, (5, 111))


# In[68]:


# Create the basis for b
X = np.vstack((np.ones(sunspot_counts[years<last_year].shape), terms[0], terms[1], terms[2], terms[3], terms[4])).T

# Nothing fancy for outputs.
Y = republican_counts[years<last_year]

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_sun = np.linspace(0, 160, 111)
grid_X = np.vstack((np.ones(grid_sun.shape), grid_terms[0], grid_terms[1], grid_terms[2], 
                    grid_terms[3], grid_terms[4]))

grid_Yhat  = np.dot(grid_X.T, w)

vec = [  0,   9,  18,  27,  36,  45,  54,  63,  72,  81,  90,  99, 108]
y_hat = []
for i in range(0,len(grid_sun)):
    if i in vec:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares ~", sum((y_hat-republican_counts[years<last_year])**2))

# Plot the data and the regression line.
plt.plot(sunspot_counts, republican_counts, 'o', grid_sun, grid_Yhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ### Part d)

# In[69]:


j_vec = list(np.arange(1,25+1,1))
    
terms = []
for j in range(0, 25):
    for i in range(0, 13):
        terms.append(math.cos(sunspot_counts[years<last_year][i]/j_vec[j]))
terms = np.reshape(terms, (25, 13))


# In[70]:


j_vec = list(np.arange(1,25+1,1))
    
grid_terms = []
for j in range(0, 25):
    for i in range(0, 111):
        grid_terms.append(math.cos(grid_sun[i]/j_vec[j]))
grid_terms = np.reshape(grid_terms, (25, 111))


# In[71]:


# Create the basis for b
X = np.vstack((np.ones(sunspot_counts[years<last_year].shape), terms[0], terms[1], terms[2], terms[3], terms[4], 
               terms[5], terms[6], terms[7], terms[8], terms[9], terms[10], terms[11], terms[12], terms[13], 
               terms[14], terms[15], terms[16], terms[17], terms[18], terms[19], terms[20], terms[21], terms[22], 
               terms[23], terms[24])).T

# Nothing fancy for outputs.
Y = republican_counts[years<last_year]

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_sun = np.linspace(0, 160, 111)
grid_X = np.vstack((np.ones(grid_sun.shape), grid_terms[0], grid_terms[1], grid_terms[2], grid_terms[3], 
                    grid_terms[4], grid_terms[5], grid_terms[6], grid_terms[7], grid_terms[8], grid_terms[9], 
                    grid_terms[10], grid_terms[11], grid_terms[12], grid_terms[13], grid_terms[14], grid_terms[15], 
                    grid_terms[16], grid_terms[17], grid_terms[18], grid_terms[19], grid_terms[20], grid_terms[21], 
                    grid_terms[22], grid_terms[23], grid_terms[24]))

grid_Yhat  = np.dot(grid_X.T, w)

vec = [  0,   9,  18,  27,  36,  45,  54,  63,  72,  81,  90,  99, 108]
y_hat = []
for i in range(0,len(grid_sun)):
    if i in vec:
        y_hat.append(grid_Yhat[i])
    else:
        pass
y_hat = np.array(y_hat)
print("Residual sum of squares ~", sum((y_hat-republican_counts[years<last_year])**2))

# Plot the data and the regression line.
plt.plot(sunspot_counts, republican_counts, 'o', grid_sun, grid_Yhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()


# ### Write Up
# The linear model here produces the lowest residual sum of squares, by quite a large margin. The two cosine models do a very bad job when it comes to fitting the number of republicans in Congress as a function of count of sunspots. I think that we could have said fairly confidently before this exercise that the number of republicans in Congress doesn't control the number of sunspots, and this exercise has certainly not shaken that for me. I'm locking in my answer of "No".
