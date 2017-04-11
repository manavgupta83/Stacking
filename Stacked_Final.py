
# coding: utf-8

# In[61]:


"""Kaggle competition: Predicting a Biological Response.
Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)
The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)
Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.
Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# def logloss(attempt, actual, epsilon=1.0e-15):
#     """Logloss, i.e. the score of the bioresponse competition.
#     """
#     attempt = np.clip(attempt, epsilon, 1.0-epsilon)
#     return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


##LOAD THE TRAINING AND TEST DATASET
data_train = pd.read_csv('train_mnist.csv')
data_test = pd.read_csv('test_mnist.csv')

##convert dataframe into array
X = data_train.ix[:,1:].as_matrix()
y = data_train.ix[:,0].as_matrix()
X_submission = data_test.as_matrix()
    
##count number of features
num_labels = len(set(y))

np.random.seed(0) # seed to shuffle the train set
idx = np.random.permutation(y.size)
X = X[idx]
y = y[idx]

#####ACTUAL START OF THE STACKED GENERALIZATION
if __name__ == '__main__':

    n_folds = 10
    verbose = True
    shuffle = False

##DEFINE THE K FOLDS
    skf = list(StratifiedKFold(y, n_folds))

##DEFINE LIST OF CLASSIFIERS
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy')
#             ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
#             ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ]

    print "Creating train and test sets for blending."

##CREATE THE SIZE OF BLENDING TRAINING AND TEST DATA
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)*num_labels))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)*num_labels))

    for j, clf in enumerate(clfs):
        j1 = j*num_labels
        print j1, clf
        
        #FOR EACH CLASSIFIER, CREATE A TEST DATASET WITH Kfold*num_labels COLUMNS
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)*num_labels)) 
        
        #LOOP FOR EACH Kfold
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            i1 = i*num_labels
            
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test) 
            
            #additional step to publish the accuracy of each classifier
            score_j = metrics.accuracy_score(y_test,clf.predict(X_test))
            print 'accuracy for %s = %s' % (j, score_j)
        
            #KEEP INSERTING PROBABILITY VALUES FROM EACH CLASSIFIER TO THE TRAINING DATASET
            dataset_blend_train[test, j1:j1+num_labels] = y_submission
            
            #IN THE Kfold*num_labels DATA INSERT PROBABILITY VALUES FROM EACH Kfold ITERATION OF TEST DATA
            dataset_blend_test_j[:, i1:i1+num_labels] = clf.predict_proba(X_submission)


        #########NOW TAKE THE AVERAGE OF ALL VALUES FOR EACH LABEL * kfold - ADDED BY MANAV GUPTA
        # for e.g. Kfolds = 5 and Num_labels = 10. For each of the Kfold we will have 10 probability outputs of actual test data
        # So for each classifier algo we will have Kfolds*Num_Labels number of variables.
        # However, we would need to get the mean of the probability values for each label across Kfolds.
        # Here we are looking at, because columns are 50, the mean values of 0,10,20,30,40 as Label 1 probability
        # Similarly the mean values of 1,11,21,31,41 for Label 2 probability
        # Below is the piece of code which does that while dynamically creating the index of the columns which should be averaged
        # First loop works for every label and creates the list shown above.
        
        for label_iter in xrange(0,num_labels):
            p = [] #define a empty list
            p.append(label_iter) 
            skf_iter = 0

            while skf_iter < len(skf)-1:
                skf_iter = skf_iter+1
                p.append(label_iter+(skf_iter*num_labels))

            #this is out of while loop but still in for loop
            dataset_blend_test[:,j1+label_iter]=dataset_blend_test_j[:,p].mean(axis = 1)       


####START THE ACTUAL BLENDING EXERCISE
print
print "Blending."
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
y_submission = clf.predict(dataset_blend_test)

# y_submission[0]

#     print "Linear stretch of predictions to [0,1]"
#     y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

# print "Saving Results."
# np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')


# In[59]:


print
print "Blending."
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
y_submission = clf.predict(dataset_blend_test)

# y_submission[0]

#     print "Linear stretch of predictions to [0,1]"
#     y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

# print "Saving Results."
# np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')


# In[63]:




# In[ ]:



