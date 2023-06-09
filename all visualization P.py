#-----------------------------------------
# Elements of data visualization in Python
#-----------------------------------------

#--------------------------------
# Scatterplot with color by group
#--------------------------------

# Load the data
from sklearn.datasets import load_iris
iris = load_iris()
iris
#{'data': array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#...
#        [6.2, 3.4, 5.4, 2.3],
#        [5.9, 3. , 5.1, 1.8]]),
# 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#  ...
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

# Scatterplot
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.title('Scatterplot with color by Species')
plt.tight_layout()
plt.show()

#----
# end
#----


#-------------------------------------------
# Scatterplot with regression lines by group
#-------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np

# Convert 'iris.data' numpy array to 'iris.dataframe' pandas dataframe
# complete the iris dataset by adding species
iris = datasets.load_iris()
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species
iris

# add regression line per group Seaborn
sns.lmplot(x="sepal length (cm)", 
           y="sepal width (cm)", 
           hue="species",
           data=iris,
           height=5)
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title('Scatterplot with regression lines by Species')

#----
# end
#----

#---------------------------------------------
# Scatterplot with marginal densities by group
#---------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
from matplotlib import gridspec
from scipy import stats

# Convert 'iris.data' numpy array to 'iris.dataframe' pandas dataframe
# complete the iris dataset by adding species
iris = datasets.load_iris()
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species
iris

cl = ['r','g', 'b'] # Custom list of colours for each categories - increase as needed...

headers = list(iris.columns) # Extract list of column headers
# Find min and max values for all x (= col [0]) and y (= col [1]) in dataframe:
xmin, xmax = iris.min(axis=0)[0], iris.max(axis=0)[0]
ymin, ymax = iris.min(axis=0)[1], iris.max(axis=0)[1]
# Create a list of all unique categories which occur in the right hand column (ie index '2'):
category_list = iris.iloc[:,5].unique()
category_list

# Set up 4 subplots and aspect ratios as axis objects using GridSpec:
gs = gridspec.GridSpec(2, 2, width_ratios=[6,2], height_ratios=[2,6])
# Add space between scatter plot and KDE plots to accommodate axis labels:
gs.update(hspace=0.3, wspace=0.3)

fig = plt.figure() # Set background canvas colour to White instead of grey default
fig.patch.set_facecolor('white')

ax = plt.subplot(gs[1,0]) # Instantiate scatter plot area and axis range
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(headers[0], fontsize = 10)
ax.set_ylabel(headers[1], fontsize = 10)
ax.yaxis.labelpad = 10 # adjust space between x and y axes and their labels if needed

axl = plt.subplot(gs[1,1], sharey=ax) # Instantiate left KDE plot area
axl.get_xaxis().set_visible(False) # Hide tick marks and spines
axl.get_yaxis().set_visible(False)
axl.spines["right"].set_visible(False)
axl.spines["top"].set_visible(False)
axl.spines["bottom"].set_visible(False)

axb = plt.subplot(gs[0,0], sharex=ax) # Instantiate bottom KDE plot area
axb.get_xaxis().set_visible(False) # Hide tick marks and spines
axb.get_yaxis().set_visible(False)
axb.spines["right"].set_visible(False)
axb.spines["top"].set_visible(False)
axb.spines["left"].set_visible(False)

axc = plt.subplot(gs[0,1]) # Instantiate legend plot area
axc.axis('off') # Hide tick marks and spines

# For each category in the list...
for n in range(0, len(category_list)):
# Create a sub-table containing only entries matching current category:
    st = iris.loc[iris[headers[5]] == category_list[n]]
    # Select first two columns of sub-table as x and y values to be plotted:
    x = st[headers[0]]
    y = st[headers[1]]

    # Plot data for each categorical variable as scatter and marginal KDE plots:    
    ax.scatter(x,y, color='none', s=100, edgecolor= cl[n], label = category_list[n], alpha = 0.4)

    kde = stats.gaussian_kde(x)
    xx = np.linspace(xmin, xmax, 1000)
    axb.plot(xx, kde(xx), color=cl[n], alpha = 0.4)

    kde = stats.gaussian_kde(y)
    yy = np.linspace(ymin, ymax, 1000)
    axl.plot(kde(yy), yy, color=cl[n], alpha = 0.4)

# Copy legend object from scatter plot to lower left subplot and display:
# NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label 
handles, labels = ax.get_legend_handles_labels()
axc.legend(handles, labels, title = headers[5], scatterpoints = 1, loc = 'upper right', fontsize = 9)

plt.show()

#----
# end
#----


#-----------------------------------------------------------
# Scatterplot with marginal densities by group using seaborn
#-----------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np


# Convert 'iris.data' numpy array to 'iris.dataframe' pandas dataframe
# complete the iris dataset by adding species
iris = datasets.load_iris()
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species
iris

sns.jointplot(data=iris, x="sepal length (cm)", y="sepal width (cm)", hue="species")

#----
# end
#----

#------------------------------------------------------------------
# Scatterplot with marginal densities and regression lines by group
#------------------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np


# Convert 'iris.data' numpy array to 'iris.dataframe' pandas dataframe
# complete the iris dataset by adding species
iris = datasets.load_iris()
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species
iris

pal = sns.color_palette(n_colors=3)
g = sns.jointplot(data=iris, x='sepal length (cm)', y='sepal width (cm)', hue='species', hue_order=['setosa', 'versicolor', 'virginica'], palette=pal)

for species, color in zip(['setosa', 'versicolor', 'virginica'], pal):
    sns.regplot(data=iris[iris['species'] == species], x='sepal length (cm)', y='sepal width (cm)', color=color, truncate=False, ax=g.ax_joint)
plt.show()

#----
# end
#----

#-----------------
# Time series plot
#-----------------


import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd

#define data
df = pd.DataFrame({'date': np.array([datetime.datetime(2020, 1, i+1)
                                     for i in range(12)]),
                   'sales': [3, 4, 4, 7, 8, 9, 14, 17, 12, 8, 8, 13]})

df2 = pd.DataFrame({'date': np.array([datetime.datetime(2020, 1, i+1)
                                      for i in range(12)]),
                   'returns': [1, 1, 2, 3, 3, 3, 4, 3, 2, 3, 4, 7]})

#plot both time series
plt.figure(figsize=(5, 4))
plt.plot(df.date, df.sales, label='sales', linewidth=3)
plt.plot(df2.date, df2.returns, color='red', label='returns', linewidth=3)
plt.title('Sales by Date')
plt.xlabel('Date',fontsize=8)
plt.ylabel('Sales', fontsize=8)
plt.show() 

#----
# end
#----

#----------------------------------------
# Histogram with Kernel Density Estimator
#----------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Inverse CDF function
def Inverse_CDF_Weibull(n, alpha, beta) :
        u = np.random.uniform(low=0.0, high=1.0, size=n)          # generate uniform numbers
        data = beta*((-np.log(1-u))**(1/alpha))                   # forumla derived
        
        return pd.DataFrame(data = data, columns = ['data'])      # return a data frame instead of an arry

# realizations and plot
np.random.seed(2023)
dataset = Inverse_CDF_Weibull(n = 1000, alpha = 5, beta = 2)
dataset
#   	data
#0	1.655497
#1	2.343973
#2	1.952544
#3	1.340683
#4	1.372833
#...	...
#995	2.069098
#996	2.056594
#997	1.996904
#998	2.001000
#999	1.925175

# plot
plot = sns.histplot(dataset, kde = True, bins = 20, facecolor="darkred", edgecolor='black')
plot.set(title='Histogram of Weibull(5,2) realizations')
plot.set(xlabel="value")
plot

#----
# end
#----

#------------------
# Multiple boxplots
#------------------

# on complete iris dataframe

sns.set(style="ticks", palette="pastel")
f, axes = plt.subplots(2, 2, sharey=False, figsize=(8, 6))
f, axes = plt.subplots(2, 2, sharey=False, figsize=(8, 6))
sns.boxplot(x="species", y="petal length (cm)",data=iris, ax = axes[0,0])
sns.boxplot(x="species", y="sepal length (cm)", data=iris, ax=axes[0,1])
sns.boxplot(x="species", y="petal width (cm)",hue = "species",data=iris, ax=axes[1,0])
sns.boxplot(x="species", y="sepal width (cm)", data=iris, ax=axes[1,1])
# adding a title to the plot
f.suptitle("Boxplot on iris dataset")
plt.show()

#----
# end
#----