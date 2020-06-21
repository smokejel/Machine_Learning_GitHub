import tensorflow as tf
import pandas as pd

# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe
def load_data_URL(file_path, URL):
    path = tf.keras.utils.get_file(file_path, URL)
    return path

def data_summary(dataset):
    # shape
    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('Species').size())

def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Define some constants
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = "/Users/michaelsweatt/PycharmProjects/Machine_Learning/Linear_Regression/iris_training.csv"
train_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
test_path = "/Users/michaelsweatt/PycharmProjects/Machine_Learning/Linear_Regression/iris_test.csv"
test_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

# train_path = load_data_URL(train_path, train_URL)
# test_path = load_data_URL(test_path, test_URL)

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# data_summary(train)
# data_summary(test)

train_y = train.pop('Species')
test_y = test.pop('Species')

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# We include a lambda to avoid creating an inner function previously
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=20000)
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))