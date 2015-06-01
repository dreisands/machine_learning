# Trey Sands
# dreisands

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import re
import random
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, f1_score
import time

def main():
	train = import_data('./data/cs-training.csv')
	train = drop_columns(train, [train.columns[0]])
	train.columns = [camel_to_snake(col) for col in train.columns]
	
	
	features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
            'monthly_income', 'age', 'number_of_times90_days_late']
	
	classifiers = 	{'LogisticRegression': {'class': LogisticRegression}, 
					'KNeighborsClassifier': {'class': KNeighborsClassifier}, 
					'DecisionTreeClassifier': {'class': DecisionTreeClassifier},
					'RandomForestClassifier': {'class': RandomForestClassifier}, 
					'AdaBoostClassifier': {'class': AdaBoostClassifier}, 
					'BaggingClassifier': {'class': BaggingClassifier}
					}

	evals = {'accuracy_score': accuracy_score, 
			'precision_score': precision_score, 
			'recall_score': recall_score, 
			'f1_score': f1_score, 
			'roc_auc_score': roc_auc_score, 
			'precision_recall_curve': precision_recall_curve}
	
	#Creating lists to loop over for parameters
	for i in range(10):
		temp = classifiers['KNeighborsClassifier'].get('kwords_list', [])
		temp.append({'n_neighbors': i})
		classifiers['KNeighborsClassifier']['kwords_list'] = temp
	for i in range(1,6,1):
		temp = classifiers['DecisionTreeClassifier'].get('kwords_list', [])
		temp.append({'max_depth': i})
		classifiers['DecisionTreeClassifier']['kwords_list'] = temp
	for i in range(2,22,2):
		temp = classifiers['RandomForestClassifier'].get('kwords_list', [])
		temp.append({'n_estimators': i})
		classifiers['RandomForestClassifier']['kwords_list'] = temp
	for i in range(50, 110, 10):
		temp = classifiers['AdaBoostClassifier'].get('kwords_list', [])
		temp.append({'n_estimators': i})
		classifiers['AdaBoostClassifier']['kwords_list'] = temp
	for i in range(6, 16, 2):
		temp = classifiers['BaggingClassifier'].get('kwords_list', [])
		temp.append({'n_estimators': i})
		classifiers['BaggingClassifier']['kwords_list'] = temp

	classifiers['LogisticRegression']['kwords_list'] = [{'C': 1.0}]
	
	x_data = train[features]
	y_data = train['serious_dlqin2yrs']
	y_pred, eval_scores, y_pred_proba = clf_cv_loop(classifiers, x_data, y_data)
	
	eval_clfs(y_pred, y_data, evals, classifiers, eval_scores, y_pred_proba)

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
	y_pred_proba = {}
	for i, j in classifiers.iteritems():
		poss_class_y_pred = []
		poss_times = []
		poss_class_y_pred_prob =[]
		for k in j['kwords_list']:
			t0 = time.time()
			temp_1, temp_2 = run_cv(x_data, y_data, j['class'], k)
			poss_class_y_pred.append(temp_1)
			poss_class_y_pred_prob.append(temp_2)
			t1 = time.time()
			total = t1-t0
			poss_times.append(total)
		y_pred[i] = poss_class_y_pred
		eval_scores[i] = {'time': poss_times}
		y_pred_proba[i] = poss_class_y_pred_prob
	return y_pred, eval_scores, y_pred_proba

def run_cv(x, y, clf_class, *args, **kwargs):
	# Construct a kfolds object
	kf = KFold(len(y),n_folds=5,shuffle=True)
	y_pred = y.copy()
	y_pred_proba = y.copy()
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
		y_pred_proba[test_index] = clf.predict_proba(x_test)
	return y_pred, y_pred_proba

def eval_clfs(y_pred, y_data, evals, classifiers, eval_scores, y_pred_proba):
	for i, j in y_pred.iteritems():
		f = open('./output/'+i+'_evals_table.csv', 'w')
		f.write('parameters\ttime\t')
		for k, l in evals.iteritems():
			f.write(k+'\t')
		f.write('precisions\trecalls\tthresholds\t')
		f.write('\n')
		for k in range(len(j)):
			f.write(str(classifiers[i]['kwords_list'][k])+'\t')
			f.write(str(eval_scores[i]['time'][k])+'\t')
			for l, m in evals.iteritems():
				if l == 'precision_recall_curve':
					precision, recall, thresholds = m(y_data, y_pred_proba[i][k])
					plt.clf()
					plt.plot(recall, precision, label='Precision-Recall curve')
					plt.xlabel('Recall')
					plt.ylabel('Precision')
					plt.ylim([0.0, 1.05])
					plt.xlim([0.0, 1.0])
					plt.title('Precision-Recall '+i+' '+ str(classifiers[i]['kwords_list'][k]))
					plt.legend(loc="lower left")
					plt.savefig('./output/'+i+'_'+str(classifiers[i]['kwords_list'][k])+'.png')
					f.write(str(precision)+'\t'+str(recall)+'\t'+str(thresholds)+'\t')
				else:
					eval_temp = m(y_data, j[k])
					f.write(str(eval_temp)+'\t')
			f.write('\n')
		f.close()

main()