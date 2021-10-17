import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import pdb

# logistic function
logistic = lambda z: 1./ (1 + np.exp(-z))

class TrainingResults:
        
        def __init__(self, lr_model, batch_size, acc_list_it, acc_list_time, acc_list_epoch, grad_list_it, grad_list_time, grad_list_epoch, final_grad_norm, total_iter, time):
            self.lr_model = lr_model
            self.batch_size = batch_size
            self.acc_list_it = acc_list_it
            self.acc_list_time = acc_list_time
            self.acc_list_epoch = acc_list_epoch
            self.grad_list_it = grad_list_it
            self.grad_list_time = grad_list_time
            self.grad_list_epoch = grad_list_epoch
            self.final_grad_norm = final_grad_norm
            self.total_iter = total_iter
            self.time = time
            
        def __repr__(self):
            return f'''\nBatch size: {self.batch_size}\n
            Weights: {self.lr_model.w}\n
            Final gradient norm: {self.final_grad_norm}\n
            Elapsed Time: {self.time}\n
            Total iterations: {self.total_iter}
            Total epochs: {len(self.acc_list_epoch)}'''
            
class LogisticRegression:
    
    def __init__(self, add_bias=True, learning_rate=.1, epsilon=1e-4, max_iters=1e5, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients 
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose
        self.gradient = self.gradient # added to avoid error when parallel processing
        
    def fit(self, x, y, batch_size=-1):
        if batch_size  == -1:
            batch_size = len(x)
        start_time = time.process_time()
        if x.ndim == 1:
            x = x[:, None] # 1d to 2d
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)]) # adding a new column with N rows
        N,D = x.shape
        self.w = np.zeros(D)
        g = np.inf 
        t = 0
        # the code snippet below is for gradient descent
        # step when exceeding max iters or gradient becomes too small
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            rand_index = np.random.choice(len(x), batch_size, replace=False)
            batch_x = x[rand_index]
            batch_y = y[rand_index]
            g = self.gradient(batch_x, batch_y)
            self.w = self.w - self.learning_rate * g 
            t += 1
        
        elapsed_time = time.process_time() - start_time

        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
            print(f'time elapsed: {elapsed_time: .2f} seconds')
            print()
        return self
    
    def fit_for_vis(self, x, y, val_X, val_y, itv=1e3):
        start_time = time.process_time()
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        self.w = np.zeros(D)
        g = np.inf 
        t = 0

        acc_list = []

        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            if t % itv == 0:
                val_yh = (self.predict(val_X) > 0.5).astype('int')
                acc_list.append(accuracy_score(val_y, val_yh))
            
            g = self.gradient(x, y)
            self.w = self.w - self.learning_rate * g
            t += 1
        
        elapsed_time = time.process_time() - start_time
        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
            print(f'time elapsed: {elapsed_time: .2f} seconds')
        return acc_list

    def fit_for_vis_complex(self, x, y, val_X, val_y, itv=1e3, batch_size=-1, max_epochs=-1, momentum=0):
        pdb.set_trace()
        if batch_size  == -1:
            batch_size = len(x)
        start_time = time.process_time()
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        print(x.shape)
        self.w = np.zeros(D)
        g = np.inf 
        t = 0

        acc_list_it = []
        acc_list_time = []
        grad_list_it = []
        grad_list_time = []
        acc_list_epoch = []
        grad_list_epoch = []
        time_itv = 0
        elapsed_time = 0
        reduced_x = x
        reduced_y = y
        epoch_done = False
        num_epochs = 0
        change = 0
        if max_epochs != -1:
            by_iters = False
            by_epochs = True
        else:
            by_iters = True
            by_epochs = False

        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and (by_epochs or t < self.max_iters) and (by_iters or num_epochs < max_epochs) :
            if t % itv == 0:
                val_yh = (self.predict(val_X) > 0.5).astype('int')
                acc_list_it.append(accuracy_score(val_y, val_yh))
                grad_list_it.append(np.linalg.norm(g))
            if elapsed_time > 15.0 * time_itv:
                val_yh = (self.predict(val_X) > 0.5).astype('int')
                acc_list_time.append(accuracy_score(val_y, val_yh))
                grad_list_time.append(np.linalg.norm(g))
                time_itv += 1
            if batch_size < len(reduced_x):
                rand_index = np.random.choice(len(reduced_x), batch_size, replace=False)
                pdb.set_trace()
                batch_x = reduced_x[rand_index]
                batch_y = reduced_y.iloc[rand_index]
                reduced_x = np.delete(reduced_x, rand_index, axis=0)
                reduced_y = reduced_y.drop(batch_y.index)
                g = self.gradient(batch_x, batch_y)
            else:
                g = self.gradient(reduced_x, reduced_y)
                reduced_x = x
                reduced_y = y
                val_yh = (self.predict(val_X) > 0.5).astype('int')
                acc_list_epoch.append(accuracy_score(val_y, val_yh))
                grad_list_epoch.append(np.linalg.norm(g))
                num_epochs += 1
            change = momentum * change + (1 - momentum) * g
            self.w = self.w - self.learning_rate * change
            t += 1
            elapsed_time = time.process_time() - start_time
        if self.verbose:
            print(f'learning rate: {self.learning_rate}')
            print(f'batch size: {batch_size}')
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
            print(f'time elapsed: {elapsed_time: .2f} seconds')
            print()
            
        result = TrainingResults(self, batch_size, acc_list_it, acc_list_time, acc_list_epoch, grad_list_it, grad_list_time, grad_list_epoch, np.linalg.norm(g), t, elapsed_time)

        return result

    def gradient(self, x, y):
        N,D = x.shape
        yh = logistic(np.dot(x, self.w))    # predictions  size N
        grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
        return grad                         # size D

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            #predict output
        return yh

