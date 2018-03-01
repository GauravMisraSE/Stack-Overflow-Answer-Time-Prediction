
# coding: utf-8

# In[125]:


import matplotlib.pyplot as plt
from math import fabs
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score, mean_squared_error, mean_squared_log_error
import numpy as np

# data to plot
n_groups = 3

#Individual obtained accuracy and mean squared log errors values of KNN, SVR and PassiveAggressive regression respectively and used below for generating a graph
accuracy = (65.2, 79.65, 80.50)
mean_squared_log_error = (6.55, 7.93, 6.99)


# create plot
fig, ax = plt.subplots(figsize=(9,6))
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
 
rects1 = plt.bar(index, accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')


rects2 = plt.bar(index + bar_width, mean_squared_log_error, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Mean-squared-log-error')

plt.xlabel('Parameters')
plt.ylabel('Percentage')
plt.title('Performance Graph')
plt.xticks(index + bar_width, ('KNN Regressor', 'Linear SVR', 'Passive Aggressive Regressor'))
plt.legend(loc = 'upper right')
 
plt.tight_layout()
plt.show()

