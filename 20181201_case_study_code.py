# ===========================================================
# Import the modules needed for this analysis
# ===========================================================
import pandas
import datetime
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# ===========================================================
# Define constants
# ===========================================================
_MTH_MAP = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
_GRD_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
_COL_ALL = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade', 'annual_inc', 'issue_d', 'dti', 'revol_bal', 'total_pymnt', 'loan_status']
_COL_NUM = ['loan_amnt', 'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_bal', 'total_pymnt']
_COL_STR = list(set(_COL_ALL) - set(_COL_NUM))

# ===========================================================
# Function to load the raw data
# ===========================================================
def load_raw_data(nrows = None):

    # Load the raw data
    if nrows:
        data = pandas.read_csv(filepath_or_buffer = r'/Users/tejsvirai/Desktop/data/loan.csv', nrows = nrows)
    else:
        data = pandas.read_csv(filepath_or_buffer = r'/Users/tejsvirai/Desktop/data/loan.csv')

    # Reindex to the requested subset of columns
    data = data.reindex(columns = _COL_ALL)

    # Return to the user
    return data

# ===========================================================
# Function to pre-process the data
# ===========================================================
def augment_data(data):

    # Remove leading and trailing zeros from the loan term
    term = list(map(lambda x: np.int(x.strip().split(' ')[0]), data['term'].values))

    # Map the grades from letters to numbers
    qual = list(map(lambda x: _GRD_MAP[x], data['grade'].values))

    # Convert the issue_d into a pandas datetime
    issue = list(map(lambda x: x.split('-'), data['issue_d']))
    issue = list(map(lambda x: datetime.datetime(np.int(x[1]), _MTH_MAP[x[0]], 1), issue))
    issue = list(map(lambda x: x + pandas.tseries.offsets.BMonthEnd(0), issue))

    # Compute the maturity date from the issue date
    expr = zip(issue, term)
    expr = list(map(lambda x: x[0] + pandas.tseries.offsets.BMonthEnd(x[1]), expr))

    # Add in dummies for whether the loan was paid off or not
    dummy  = [1.0] * len(data.index)
    isPaid = list(map(lambda x: 1.0 if 'fully paid' in x.lower() else 0.0, data['loan_status'].values))
    yyyy   = list(map(lambda x: x.year, issue))
    annRet = (data['total_pymnt'].values / data['funded_amnt'].values) ** (1 / 3) - 1.0

    # Store the data
    data['term_mm']  = term
    data['grd_num']  = qual
    data['issue_dt'] = issue
    data['expr_dt']  = expr
    data['dummy']    = dummy
    data['ispaid']   = isPaid
    data['yyyy']     = yyyy
    data['avgRet']   = annRet

    # Return the data to the user
    return data

# ===========================================================
# Function to visualize the data
# ===========================================================
def visualize_data(data):

    # Print a subset of the data to see what it looks like
    print('Printing a subset of the data')
    print('=============================')
    print(data.tail(50).to_string())
    print('=============================')
    print('')

    # Print a summary of the data to see how many missing values / the data types of the different columns
    # For example, is the issue_d date a date or an object?
    print('Printing column data types for inspection')
    print('=========================================')
    print(data.info())
    print('=========================================')
    print('')

    # Test for outliers in the numeric data. Specifically, we want to see the following:
    # A: loan_amnt   >  0
    # B: funded_amnt >  0
    # C: int_rate    >  0 and < something large like, say, 100%
    # D: annual_inc  >= 0
    # E: dti         >= 0 and < something large like, say, 10x
    # F: revol_bal   >= 0
    # G: total_pymt  >= 0
    # H: B/A ratio   >= 0 and <= 1
    print('Printing summary statistics for numerical columns')
    print('=================================================')
    data['funded_to_loan_ratio'] = data['funded_amnt'] / data['loan_amnt']
    print(data.describe().to_string())
    print('=================================================')
    print('')

    # Check for the list of values possible in the non-numeric data
    print('Printing unique values for string columns')
    print('==========================================')
    for c in _COL_STR:
        print('Column ' + c + ': ' + str(sorted(list(set(data[c])))))
    print('==========================================')

    # Plot values of the numeric columnss
    for c in _COL_NUM:
        plt.figure()
        _data = data.reindex(columns = [c])
        _data = _data.sort_values(by = [c])
        _data = pandas.DataFrame(columns = [c], data = _data.values)
        _data.plot()
        plt.show()
        _data.plot.hist(bins = 20)
        plt.show()

    # Examine the individual columns that seem to have bad data
    print('Examining individual errors one by one')
    print('======================================')
    for c in ['annual_inc', 'dti', 'revol_bal']:
        print(c)
        _data = data.sort_values(by = [c])
        print(_data.head(25).to_string())
        print(_data.tail(25).to_string())
    print('======================================')
    print('')

# ===========================================================
# Function to clean the data
# ===========================================================
def clean_data(data):

    # Set DTI to missing if self-reported income is zero
    data.loc[data['annual_inc'] == 0.0, 'dti'] = np.NaN

    # Winsorize annual_inc, dti and revol_bal at the 99.9th percentile to avoid huge skew issues
    for c in ['annual_inc', 'dti', 'revol_bal']:
        quantile = data[c].quantile([0.999]).values[0]
        data.loc[data[c] >= quantile, c] = quantile

    # Return data to the user
    return data

# ===========================================================
# Function to analyze aggregations of the data
# ===========================================================
def analyze_data(data):

    for key in [['yyyy'], ['grade'], ['term'], ['yyyy', 'grade'], ['yyyy', 'term']]:
        print('Analyzing aggregations of data with key: ' + str(key))
        print('===============================')
        group = data.groupby(by = key)
        group = group.agg({'dummy': np.sum, 'int_rate': np.mean, 'loan_amnt': np.mean, 'annual_inc': np.mean, 'dti': np.mean, 'ispaid': np.mean})
        print(group.to_string())
        print('===============================')
        print('')
    return data

# ===========================================================
# Function to answer business questions
# ===========================================================
def business_analysis(data):

    # Restrict the data to only contain the 36 month loans
    data = data.loc[data['term_mm'].values == 36, :]

    # Convert the issue_d into a pandas datetime and assume that the maximum issue date is the current date
    today = max(data['issue_dt'].values)
    flag = np.array(list(map(lambda x: x <= today, data['expr_dt'].values)))
    data = data.loc[flag, :]

    # Question 1:
    fullyPaid = data['ispaid'].sum() / data['dummy'].sum()

    # Question 2:
    groups = data.groupby(by = ['yyyy', 'grade'])
    cohort = groups.sum()
    cohort = 1.0 - cohort['ispaid'] / cohort['dummy']
    cohort = cohort.sort_values()
    cohort = cohort.index[-1]

    # Question 3:
    avgRet = groups.mean()['avgRet']

    # Print answers to the questions
    print('Business Analysis Output:')
    print('================================================')
    print('Question 1: The percentage of loans that has been fully paid is: ' + str(fullyPaid * 100) + '%')
    print('Question 2: The most deliquent cohort is: ' + str(cohort))
    print('Question 3: The average return by cohort is:')
    print(avgRet.to_string())
    print('================================================')
    print('')

    # Return data to the user
    return data, fullyPaid, cohort, avgRet

# ===========================================================
# Function to run logistic regressions
# ===========================================================
def logistic_regression(data, testSize):

    # Augment data with a few key predictors
    data['lti']     = data['loan_amnt'] / data['annual_inc']
    data['dti_lti'] = data['lti'] + data['dti']

    # Drop data with NaNs
    data = data.reindex(columns = ['int_rate', 'grd_num', 'annual_inc', 'dti_lti', 'ispaid'])
    data = data.dropna(how = 'any', axis = 0)

    # Extract out some relevant features
    X = data.reindex(columns = ['int_rate', 'grd_num', 'annual_inc', 'dti_lti'])
    Y = data['ispaid']

    # Split the data set into training and test sets
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = testSize)

    # Fit the logistic regression model
    regress = LogisticRegression(class_weight = 'balanced')
    regress.fit(X = XTrain, y = YTrain)

    # Predict output on the testing sets
    YPred = regress.predict(X = XTest)

    # Generate the confusion matrix
    conf = confusion_matrix(y_true = YTest, y_pred = YPred)
    conf = pandas.DataFrame(index = ['true_0', 'true_1'], columns = ['pred_0', 'pred_1'], data = conf)
    conf = conf / conf.sum().sum()

    # Generate standardized output
    accuracy  = accuracy_score(y_true = YTest, y_pred = YPred)
    precision = precision_score(y_true = YTest, y_pred = YPred)
    recall    = recall_score(y_true = YTest, y_pred = YPred)

    # Print answers to the questions
    print('Logistic Regression Output')
    print('================================================')
    print('Accuracy:  ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall:    ' + str(recall))
    print('Confusion matrix:')
    print(conf.to_string())
    print('================================================')
    print('')

    # Return metrics to the user
    return conf, accuracy, precision, recall

# ===========================================================
# Testing area
# ===========================================================
if __name__ == "__main__":

    data = load_raw_data(nrows = None)
    data = augment_data(data = data)
    visualize_data(data = data)
    data = clean_data(data = data)
    data = analyze_data(data = data)
    data, fullyPaid, cohort, avgRet   = business_analysis(data = data)
    conf, accuracy, precision, recall = logistic_regression(data = data, testSize = 0.20)