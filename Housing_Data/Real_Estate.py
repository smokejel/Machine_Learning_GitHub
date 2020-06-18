# Load Libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def data_summary(dataset):
    # shape
    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('postal_code').size())

def data_visualization(dataset):
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()

    # histograms
    dataset.hist()
    pyplot.show()

    # scatter plot matrix
    scatter_matrix(dataset)
    pyplot.show()

def split_dataset(dataset):
    # Split-out validation dataset
    array = dataset.values
    print(array)
    X = array[:, 7:10]
    y = array[:, 10]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    return X_train, X_validation, Y_train, Y_validation

def check_models(X_train, Y_train):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Compare Algorithms
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()

def predictions(X_train, X_validation, Y_train, Y_validation):
    # Make predictions on validation dataset
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

# Load dataset
names = ['month_date_yyyymm', 'postal_code', 'zip_name', 'nielsen_hh_rank', 'hotness_rank', 'hotness_rank_mm',
         'hotness_rank_yy', 'hotness_score', 'supply_score', 'demand_score', 'median_days_on_market',
         'median_days_on_market_mm', 'median_days_on_market_mm_day', 'median_days_on_market_yy',
         'median_days_on_market_yy_day', 'median_days_on_market_vs_us', 'ldpviews_per_property_mm',
         'ldpviews_per_property_yy', 'ldpviews_per_property_vs_us', 'median_listing_price', 'median_listing_price_mm',
         'median_listing_price_yy', 'median_listing_price_vs_us']
dataset = read_csv('RDC_Inventory_Hotness_Metrics_Zip_History.csv', names=names)

# data_summary(dataset)
# data_visualization(dataset)
X_train, X_validation, Y_train, Y_validation = split_dataset(dataset)
check_models(X_train, Y_train)
# predictions(X_train, X_validation, Y_train, Y_validation)
