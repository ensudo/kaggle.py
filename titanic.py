import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.utils import shuffle
from torch.autograd import Variable


# TODO incorporate one hot encoding on the classes (survived)... matrices have to be of a single type can't combine matrices and floats

min_max_scaler = preprocessing.MinMaxScaler()

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

passenger = pd.read_csv("passenger.csv")
survived = pd.read_csv('survived.csv')

"""
    survival - categorical
    sex - categorical
    ticket class - ordinal
    age - continuous
    siblings/spouses - count
    parents/children -count
    ticket - etc, etc :)
    
"""
passenger.describe(include='all')

# replace null with modal value
modal_sex = passenger['Sex'].mode()
passenger['Sex'].fillna(modal_sex, inplace=True)

# replace null with mean age
mean_age = passenger['Age'].mean()
passenger['Age'].fillna(mean_age, inplace=True)

# replace null with mean fare
mean_fare = passenger['Fare'].mean()
passenger['Fare'].fillna(mean_fare, inplace=True)

features = passenger[['PassengerId', 'Sex', 'SibSp', 'Age', 'Parch', 'Fare', 'Pclass']].to_numpy(dtype='object')

feature_label_encoder = preprocessing.LabelEncoder()

for i in range(1, 7):   # except PassengerId
    features[:, i] = feature_label_encoder.fit_transform(features[:, i])

features = preprocessing.normalize(features, axis=1, norm='l1')

# one-hot code sex
# sex_one_hot_encoder = preprocessing.OneHotEncoder(categories=[np.unique(features[:, 0])])
# features[:, 0] = sex_one_hot_encoder.fit_transform(features[:, 0].reshape(-1, 1))

classes = survived[['Survived', 'PassengerId']].to_numpy()

classes_label_encoder = preprocessing.LabelEncoder()

classes[:, 0] = classes_label_encoder.fit_transform(classes[:, 0])
classes[:, 1] = classes_label_encoder.fit_transform(classes[:, 1])

# one-hot code survived
# survived_one_hot_encoder = preprocessing.OneHotEncoder(categories=[np.unique(classes[:, 0])])
# classes[:, 0] = survived_one_hot_encoder.fit_transform(classes[:, 0].reshape(-1, 1))

classes = survived[['Survived', 'PassengerId']].to_numpy()

x_train, x_test, y_train, y_test = model_selection.train_test_split(features, classes[:, 0], test_size=0.2,
                                                                    random_state=1)

print(np.shape(features))

print(np.shape(x_train))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 270)
        self.fc2 = nn.Linear(270, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


net = Net()

batch_size = 50
num_epochs = 200
learning_rate = 0.01
batch_no = len(x_train) // batch_size

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('Epoch {}'.format(epoch + 1))
    x_train, y_train = shuffle(x_train, y_train)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(x_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end]))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        ypred_var = net(x_var)
        loss = criterion(ypred_var, y_var)
        loss.backward()
        optimizer.step()

test_var = Variable(torch.FloatTensor(x_test), requires_grad=True)
with torch.no_grad():
    result = net(test_var)
values, labels = torch.max(result, 1)
num_right = np.sum(labels.data.numpy() == y_test)
accuracy = num_right / len(y_test)
assert accuracy > 0.8
print('Accuracy {:.2f}'.format(accuracy))
