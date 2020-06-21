import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
import tensorflow as tf

def data_summary(dataset):
    # shape
    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('sex').size())

# This function is used to take a pandas data frame and transform it into a Tensor Flow Dataset Object
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

# Load Dataset
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
train_path = tf.keras.utils.get_file('/Users/michaelsweatt/PycharmProjects/Machine_Learning/Tensor_Flow/titanic_train.csv',
                                     'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
eval_path = tf.keras.utils.get_file('/Users/michaelsweatt/PycharmProjects/Machine_Learning/Tensor_Flow/titanic_eval.csv',
                                    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
dftrain = pd.read_csv(train_path)
dfeval = pd.read_csv(eval_path)

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
print(train_path)
print(eval_path)

'''Save Local Copy
dftrain.to_csv('/Users/michaelsweatt/PycharmProjects/Machine_Learning/Tensor_Flow/titanic_train.csv')
dfeval.to_csv('/Users/michaelsweatt/PycharmProjects/Machine_Learning/Tensor_Flow/titanic_eval.csv')'''

'''data_summary(dftrain)
dftrain.age.hist(bins=20)
plt.show()'''

# Create feature columns within data frame
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=float))

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# We create a linear estimator by passing the feature columns we created earlier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

# print(result['accuracy'])  # the result variable is simply a dict of stats about our model
# print(result)

mk = 2
result2 = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[mk])
print(y_eval.loc[mk])
print(result2[mk]['probabilities'][1])
