import sklearn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
import pandas as pd
from random import sample

# np.set_printoptions(threshold='nan')
class Metric(object):
    def __init__(self,output,label):
        self.output = output   #prediction label matric
        self.label = label    #true  label matric
 
    def accuracy_subset(self,threash=0.5):
        y_pred =self.output
        y_true = self.label
        y_pred=np.where(y_pred>threash,1,0)
        accuracy=accuracy_score(y_true,y_pred)
        return accuracy
    
    def accuracy(self,threash=0.5):
        y_pred =self.output
        y_true = self.label      
        y_pred=np.where(y_pred>threash,1,0)
        accuracy=sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
        return accuracy
    
    def accuracy_multiclass(self):
        y_pred =self.output
        y_true = self.label     
        accuracy=accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
        return accuracy
    
    def micfscore(self,threash=0.5,type='micro'):
        y_pred =self.output
        y_true = self.label
        y_pred=np.where(y_pred>threash,1,0)
        return f1_score(y_pred,y_true,average=type)

    def macfscore(self,threash=0.5,type='macro'):
        y_pred =self.output
        y_true = self.label
        y_pred=np.where(y_pred>threash,1,0)
        return f1_score(y_pred,y_true,average=type)
    
    def hamming_distance(self,threash=0.5):
        y_pred =self.output
        y_true = self.label
        y_pred=np.where(y_pred>threash,1,0)
        return hamming_loss(y_true,y_pred)
    
    def fscore_class(self,type='micro'):
        y_pred =self.output
        y_true = self.label
        return f1_score(np.argmax(y_pred,1),np.argmax(y_true,1),average=type)
    
    def auROC(self):
        y_pred =self.output
        y_true = self.label
        row,col = y_true.shape
        temp = []
        ROC = 0
        for i in range(col):
            try:
                ROC = roc_auc_score(y_true[:,i], y_pred[:,i], average='micro', sample_weight=None)
            except:
                ROC == 0.5  
            temp.append(ROC)
        for i in range(col):
            ROC += float(temp[i])
        return ROC/(col+1),temp
    
    def MacroAUC(self):
        y_pred =self.output #num_instance*num_label
        y_true = self.label #num_instance*num_label 
        num_instance,num_class =  y_pred.shape
        count = np.zeros((num_class,1))   # store the number of postive instance'score>negative instance'score
        num_P_instance =  np.zeros((num_class,1)) #number of positive instance for every label      
        num_N_instance =  np.zeros((num_class,1)) 
        auc = np.zeros((num_class,1))  # for each label
        count_valid_label = 0
        for  i in range(num_class): 
            num_P_instance[i,0] = sum(y_true[:,i] == 1) #label,,test_target
            num_N_instance[i,0] = num_instance - num_P_instance[i,0]
            # exclude the label on which all instances are positive or negative,
            # leading to num_P_instance(i,1) or num_N_instance(i,1) is zero
            if num_P_instance[i,0] == 0 or num_N_instance[i,0] == 0:
                auc[i,0] = 0
                count_valid_label = count_valid_label + 1
            else:
                temp_P_Outputs = np.zeros((int(num_P_instance[i,0]), num_class))
                temp_N_Outputs = np.zeros((int(num_N_instance[i,0]), num_class))
                #
                temp_P_Outputs[:,i] = y_pred[y_true[:,i]==1,i]
                temp_N_Outputs[:,i] = y_pred[y_true[:,i]==0,i]    
                for m in range(int(num_P_instance[i,0])):
                    for n in range(int(num_N_instance[i,0])):
                        if(temp_P_Outputs[m,i] > temp_N_Outputs[n,i] ):
                            count[i,0] = count[i,0] + 1
                        elif(temp_P_Outputs[m,i] == temp_N_Outputs[n,i]):
                            count[i,0] = count[i,0] + 0.5
                
                auc[i,0] = count[i,0]/(num_P_instance[i,0]*num_N_instance[i,0])  
        macroAUC1 = sum(auc)/(num_class-count_valid_label)
        return  float(macroAUC1), auc


def bootstrap_auc(label, output, classes, bootstraps=5, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))
    for c in range(len(classes)):
        for i in range(bootstraps):
            L=[]
            for k in range(len(label)):
                L.append([output[k],label[k]])
            if fold_size <= len(L):
                X = sample(L, fold_size)
            else:
                fold_size == len(L)
                X = sample(L, fold_size)
            for b in range(len(X)):
                if b ==0:
                    Output =  np.array([X[b][0]])
                    Label =  np.array([X[b][1]])
                Output = np.concatenate((Output, np.array([X[b][0]])),axis=0)
                Label = np.concatenate((Label, np.array([X[b][1]])),axis=0)
                
            myMetic = Metric(Output,Label)
            AUROC1, auc = myMetic.auROC()
            statistics[c][i] = AUROC1
    return statistics
    


