# Trey Sands

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import re
import random
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import LinearSVC as LSVC
from sklearn.ensemble import RandomForestClassifier as RFC 
from sklearn.ensemble import AdaBoostClassifier as ABC 
from sklearn.ensemble import BaggingClassifier as BC 
from sklearn.metrics import accuracy_score as AS, precision_score as PS 
from sklearn.metrics import recall_score as RS, precision_recall_curve as PRC
from sklearn.metrics import roc_auc_score as RAS, f1_score as FS
import time

def main():
	train = import_data('./data/cs-training.csv')
	train = drop_columns(train, [train.columns[0]])
	train.columns = [camel_to_snake(col) for col in train.columns]
	
	explore_data(train)
	make_hist(train, column = 'revolving_utilization_of_unsecured_lines', 
		bins = 3)
	make_hist(train, column = 'age', bins = 10)
	make_hist(train, column = 'number_of_dependents', bins = 10)
	make_hist(train, column = 'number_of_open_credit_lines_and_loans'
		, bins = 10)
	
	features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
            'monthly_income', 'age', 'number_of_times90_days_late']
	
	classifiers = {'LR': {'class': LR}, 'KNC': {'class': KNC}, 
		'DTC': {'class': DTC}, 'LSVC': {'class': LSVC}, 'RFC': {'class': RFC}, 
		'ABC': {'class': ABC}, 'BC': {'class': BC}}

	#classifiers = {'DTC': {'class': DTC}}
	evals = {'AS': AS, 'PS': PS, 'RS': RS, 'FS': FS, 'RAS': RAS, 'PRC': PRC}
	
	#Creating lists to loop over for parameters
	for i in range(10):
		temp = classifiers['KNC'].get('kwords_list', [])
		temp.append({'n_neighbors': i})
		classifiers['KNC']['kwords_list'] = temp
	for i in range(1,6,1):
		temp = classifiers['DTC'].get('kwords_list', [])
		temp.append({'max_depth': i})
		classifiers['DTC']['kwords_list'] = temp
	for i in range(2,22,2):
		temp = classifiers['RFC'].get('kwords_list', [])
		temp.append({'n_estimators': i})
		classifiers['RFC']['kwords_list'] = temp
	for i in range(50, 110, 10):
		temp = classifiers['ABC'].get('kwords_list', [])
		temp.append({'n_estimators': i})
		classifiers['ABC']['kwords_list'] = temp
	for i in range(6, 16, 2):
		temp = classifiers['BC'].get('kwords_list', [])
		temp.append({'n_estimators': i})
		classifiers['BC']['kwords_list'] = temp

	classifiers['LR']['kwords_list'] = [{'C': 1.0}]
	classifiers['LSVC']['kwords_list'] = [{'C': 1.0}]
	
	x_data = train[features]
	y_data = train['serious_dlqin2yrs']
	y_pred, eval_scores = clf_cv_loop(classifiers, x_data, y_data)
	
	eval_clfs(y_pred, y_data, evals, classifiers, eval_scores)

def import_data(file_name):
	return pd.read_csv(file_name)

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def drop_columns(data, list_of_ind):
	return data.drop(list_of_ind, axis=1)

def explore_data(data):
	f = open('./output/summary_stats.txt', 'w')

	for header in data.select_dtypes(include = [np.number]).columns.tolist():
		f.write(header + " summary statistics:")
		f.write("\nmean: " + str(data[header].mean()))
		f.write("\nmedian: " + str(data[header].median()))
		f.write("\nmode: " + str(data[header].mode()))
		f.write("\nstd: " + str(data[header].std()))
		f.write("\n# of missing values: " + str(pd.isnull(data[header]).sum()))
		f.write("\n\n")

	for header in data.select_dtypes(exclude = [np.number]).columns.tolist():
		f.write(header + " summary statistics:")
		f.write("\nmode: " + str(data[header].mode()))
		f.write("\n# of missing values: " + str(pd.isnull(data[header]).sum()))
		f.write("\n\n")

	f.close()

def make_hist(data, column=None, bins=10):
	temp = plt.figure()
	data[column].plot(kind = 'hist', bins=bins)
	plt.xlabel(column)
	plt.ylabel("Frequency")
	plt.savefig('./output/'+column+'_hist.png')
	plt.close(temp)

def create_ktiles(data, col_name, k):
	bins = []

	for q in range(1,k+1):
		bins.append(data[col_name].quantile(1.0/k*q))

	data[col_name+'_bins'] = pd.cut(data[col_name], bins = bins)

def create_bins(data, col_name, n):
	data[col_name+'_bins'] = pd.cut(data[col_name], bins = n, labels = False)

def create_binary_from_cat(data, new_col_name, col_name, one):
	data[new_col_name] = data[col_name] == one

def clf_cv_loop(classifiers, x_data, y_data):
	y_pred = {}
	eval_scores = {}
	for i, j in classifiers.iteritems():
		poss_class = []
		poss_times = []
		for k in j['kwords_list']:
			t0 = time.time()
			poss_class.append(run_cv(x_data, y_data, j['class'], k))
			t1 = time.time()
			total = t1-t0
			poss_times.append(total)
		y_pred[i] = poss_class
		eval_scores[i] = {'time': poss_times}
	return y_pred, eval_scores

def run_cv(x, y, clf_class, *args, **kwargs):
	# Construct a kfolds object
	kf = KFold(len(y),n_folds=5,shuffle=True)
	y_pred = y.copy()
	# Iterate through folds
	for train_index, test_index in kf:
		x_train = x.ix[train_index]
		x_test  = x.ix[test_index]
		y_train = y.ix[train_index]
		x_train = Imputer(strategy = 'median').fit_transform(x_train)
		x_test = Imputer(strategy = 'median').fit_transform(x_test)
		# Initialize a classifier with key word arguments
		clf = clf_class(**kwargs)
		clf.fit(x_train,y_train)
		y_pred[test_index] = clf.predict(x_test)
	return y_pred

def eval_clfs(y_pred, y_data, evals, classifiers, eval_scores):
	for i, j in y_pred.iteritems():
		f = open('./output/'+i+'_evals_table.csv', 'w')
		f.write('parameters\ttime\t')
		for k, l in evals.iteritems():
			f.write(k+'\t')
		f.write('\n')
		for k in range(len(j)):
			f.write(str(classifiers[i]['kwords_list'][k])+'\t')
			f.write(str(eval_scores[i]['time'][k])+'\t')
			for l, m in evals.iteritems():
				clf_temp = eval_scores.get(i)
				eval_temp = m(y_data, j[k])
				kword_temp = clf_temp.get(l,[])
				kword_temp.append(eval_temp)
				clf_temp[l] = kword_temp
				eval_scores[i] = clf_temp
				f.write(str(eval_temp)+'\t')
			f.write('\n')
		f.close()

main()