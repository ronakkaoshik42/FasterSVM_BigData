# set up packages
import cvxpy as cp
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle

class MyClassifier:
    # Basic constructor setting all attributes to empty/zero, feel free to change
    def __init__(self):
        self.training_set_data = np.array([])
        self.training_set_labels = np.array([])
        self.weights = []
        self.biases = []
        self.num_classes = 2
        self.num_features = 0
        self.num_queries = 0
        self.label_map = {-1:-1,1:1}
        self.delta = 0.04
        self.tune = 7

    # determines whether we should query the passed label for a data sample. Note that the selection
    # criteria cannot depend on this label. If we don't want to query, we will assign a label to the point
    def sample_selection(self,training_sample,label):
        # very simple method checks if a training sample is more than some threshold delta away from 
        Wxb = self.weights.T @ training_sample + self.biases.T
        if (abs(Wxb) > self.delta).all(): 
            # do not query, instead add the training sample to the training set with PREDICTED label from our model
            # This follows the CAL method, but not sure if this is actually valid for our goals, which is to minimize 
            # ntrain
            0
        else:
            # otherwise, query the point and add the training sample with the actual label
            self.num_queries += 1
            self.training_set_data = np.row_stack([self.training_set_data, training_sample])
            self.training_set_labels = np.append(self.training_set_labels, label)
        return self

        # Description: Calls LP relaxation, constructs integral solution, and adds 
    # relevant data points and labels to the self.training_set_* structures
    # Inputs: self, training_set (NxM), training_labels (Nx1)
    # Outputs: self, with training points selected
    def ILP(self,training_set,training_labels,Ntrain):
        # Map classes to -1 and 1
        min_lab = min(training_labels)
        max_lab = max(training_labels)
        training_labels = training_labels - min(training_labels)
        training_labels = 2*training_labels/max(training_labels)
        training_labels = training_labels - 1
        training_labels = np.array(training_labels)
        training_labels = np.reshape(training_labels,(training_labels.shape[0],1))

        # Solve LP relaxation
        lp_sol = self.LP(training_set,training_labels,Ntrain)
        self.label_map[-1] = min_lab
        self.label_map[1] = max_lab

        # Round to create integral solution
        ilp_sol = np.around(lp_sol)
        # Use integral solution to add selected points to self.training_set_*
        self.training_set_data = np.empty((training_set.shape[1],0))
        for i in range(1,ilp_sol.shape[0]):
            if ilp_sol[i]:
                self.training_set_data = np.column_stack([self.training_set_data, training_set[i,:]])
                key = training_labels[i,0]
                self.training_set_labels = np.append(self.training_set_labels, self.label_map[key])
        self.training_set_data = self.training_set_data.T
        self.num_queries = self.training_set_data.shape[0]
        return self


    # Description: Solves LP relaxation of the sample selection ILP
    # Inputs: self, training_set (NxM), training_labels (Nx1)
    # Outputs: lam, a (Nx1) variable indicating whether each point in the training_set
    # should be used for training or not
    def LP(self,training_set,training_labels,Ntrain):
        # Sets the minimum number of training points to be included from each class
        N = training_set.shape[0]

        # Seperates the classes and find the mean of each class
        indices_class1 = [i for i, x in enumerate(training_labels) if x == 1]
        indices_class2 = [i for i, x in enumerate(training_labels) if x == -1]
        class_1_avg = np.mean(training_set[indices_class1,:],axis=0)
        class_2_avg = np.mean(training_set[indices_class2,:],axis=0)

        avg_data = np.array(np.vstack([class_1_avg,class_2_avg]))
        avg_labels = np.array(np.hstack([[1], [-1]]))
        
        # lc_avg = MyClassifier()    
        self.train(avg_data, avg_labels)
        avg_W = self.weights
        avg_B = self.biases

        # Calculates the label-weighted distance between each point and the class means
        d1 = np.zeros((N,1))
        d2 = np.zeros((N,1))
        for i in range(1,N):
            avgdist = training_labels[i]*np.linalg.norm(training_set[i,:] - class_1_avg,2) - training_labels[i]*np.linalg.norm(training_set[i,:] - class_2_avg,2)
            hyperdist = avg_W.T @ training_set[i,:] + avg_B
            d1[i,0] = avgdist
            d2[i,0] = abs(hyperdist)
        # Solves the LP
        lam1 = cp.Variable((N,1),nonneg=True)
        lam2 = cp.Variable((N,1),nonneg=True)
        dlam1 = cp.multiply(lam1,d1)
        dlam2 = cp.multiply(lam2,d2)
        cost = cp.sum(dlam1)+cp.sum(dlam2)


        cons = [cp.sum(lam2) >= (self.tune)/10*Ntrain,cp.sum(cp.multiply(lam1,training_labels))+cp.sum(cp.multiply(lam2,training_labels)) == 0, -lam1 >= -1, -lam2 >= -1, cp.sum(lam1)+cp.sum(lam2) <= Ntrain, -(lam1 + lam2) >= -1]
        prob = cp.Problem(cp.Minimize(cost),cons)
        print('solving...')
        prob.solve()
        #return lam1.value
        return np.maximum(lam1.value, lam2.value)


    # Description: trains the classifier object using the data from the training set
    # returns the updated classifier object with trained weights and biases
    # Inputs: self, train_data (NxM), train_label (Nx1)
    # Outputs: updated self object with assigned self.weights and self.biases
    def train(self,train_data,train_label):
        # Update the number of features M, the number of data points N, and the 
        # number of classes
        train_data = train_data.T
        M = train_data.shape[0]
        N = train_data.shape[1]
        train_label = np.array(train_label)
        train_label = np.reshape(train_label,(train_label.shape[0],1))
        self.num_features = M
        self.num_classes = np.unique(train_label).shape[0]
        self.num_queries = N
        L = self.num_classes - 1

        # Raise exception if the problem is not a binary classification task
        if self.num_classes != 2:
            raise Exception("Training set must have exactly 2 classes.")
        else:   
            # map classes to 1 and -1.  
            self.label_map[-1] = min(train_label)
            self.label_map[1] = max(train_label)
            train_label = train_label - min(train_label)
            train_label = 2*train_label/max(train_label)
            train_label = train_label - 1
            
            # Use cpvpy to create the optimization problem
            weight_matrix = cp.Variable((M,L))
            bias_vector = cp.Variable((L,1))
            hinge_loss = cp.sum(cp.pos(1 - cp.multiply(train_label.T, weight_matrix.T @ train_data + bias_vector)))

            # Set up regularization term
            L1_reg = cp.sum(cp.norm(weight_matrix,1))
            reg_parameter = cp.Parameter(nonneg=True,value=0)

            # Solve problem and update object accordingly
            LP_prob = cp.Problem(cp.Minimize(hinge_loss/N+reg_parameter*L1_reg))
            LP_prob.solve()
            self.weights = weight_matrix.value
            self.biases = bias_vector.value
        return self


    # Description: returns a predicted class based on the sign of the input_vector
    # Inputs: self, input_vec (Lx1) equal to W^T x + w
    # Outputs: predicted class
    def f(self,input_vec):
        # Returns 1 if input_vec is positive, since that means x is above all hyperplanes
        # and -1 otherwise
        if (input_vec >= np.zeros(input_vec.shape)).all():
          return self.label_map[1]
        else:
          return self.label_map[-1]


    # Description: returns predicted classes for a set of test points
    # Inputs: self, test_set (N_testxM)
    # Outputs: predicted_classes (N_testx1)
    def test(self,test_set):
        # Iterate through each test point and run f(W^T x + w)
        predicted_classes = np.zeros(test_set.shape[0])
        for i in range(test_set.shape[0]):
          gy = self.weights.T @ test_set[i,:] + self.biases
          predicted_classes[i] = self.f(gy)
        return predicted_classes
        # Description: returns accuracy percentage for test set by comparing the 
    # predicted label to the ground truth label
    # Inputs: self, test_data (N_testxM), labels (N_testx1)
    # Outputs: percent accuracy


    def accuracy(self,test_data,labels):
        # Make sure input is correctly formatted
        labels = np.array(labels)
        labels = np.reshape(labels,(labels.shape[0],1))

        # Compare predicted labels to true labels and calculate percent accuracy
        test_labels = self.test(test_data)
        correct_labels = [1*(test_labels[i] == labels[i]) for i in range(test_data.shape[0])]
        percent = np.sum(correct_labels)/test_data.shape[0]
        return percent
        
