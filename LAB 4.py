#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)
def entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
class_entropy = entropy(df['buys_computer'])
features = df.columns[:-2] 
informationgains = {}

for feature in features:
    weighted_entropy = 0
    for value in df[feature].unique():
        subset = df[df[feature] == value]
        subset_entropy = entropy(subset['buys_computer'])
        weight = len(subset) / len(df)
        weighted_entropy += weight * subset_entropy
    informationgain = class_entropy - weighted_entropy
    informationgains[feature] = informationgain

root = max(informationgains, key=informationgains.get)

print("Information Gains:")
for feature, gain in informationgains.items():
    print(f"{feature}: {gain}")

print(f"first feature for constructing the decision tree is: {root}")


# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buyscomputer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)
df2= pd.get_dummies(df, columns=['age', 'income', 'student', 'credit_rating'])
X = df2.drop('buyscomputer', axis=1)
y = df2['buyscomputer']
model = DecisionTreeClassifier()
model.fit(X, y)
tree_depth = model.get_depth()
print("Tree Depth:", tree_depth)


# In[4]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Assuming you have already created and fitted your Decision Tree model (model)

# Set the figure size for a larger visualization
plt.figure(figsize=(90 , 40))

# Use plot_tree to visualize the Decision Tree
plot_tree(model, filled=True)

# Show the plot
plt.show()


# In[ ]:


# Import the necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Assuming you have loaded your data into variables Tr_X, Tr_y, Te_X, Te_y
df=pd.read_csv(r"C:\Users\mamid\OneDrive\Desktop\MACHINE LEARNING LAB\extracted_features_charrec.csv")
X = df.drop(columns=['class_name'])
y = df['class_name']

# Step 2: Split the data into training and test sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and fit the Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(Tr_X, Tr_y)

# Step 4: Evaluate the accuracy of the model
training_accuracy = model.score(Tr_X, Tr_y)
test_accuracy = model.score(Te_X, Te_y)

print(f"Training Set Accuracy: {training_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(model, filled=True, feature_names=list(X.columns))
plt.title("Decision Tree")
plt.show()


# In[ ]:


# Import the necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Assuming you have loaded your data into variables Tr_X, Tr_y, Te_X, Te_y
df=pd.read_csv(r"C:\Users\mamid\OneDrive\Desktop\MACHINE LEARNING LAB\extracted_features_charrec.csv")
X = df.drop(columns=['class_name'])
y = df['class_name']

# Step 2: Split the data into training and test sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and fit the Decision Tree classifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(Tr_X, Tr_y)

# Step 4: Evaluate the accuracy of the model
training_accuracy = model.score(Tr_X, Tr_y)
test_accuracy = model.score(Te_X, Te_y)

print(f"Training Set Accuracy: {training_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(model, filled=True, feature_names=list(X.columns))
plt.title("Decision Tree")
plt.show()


# In[ ]:




