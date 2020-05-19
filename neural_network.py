import numpy as np
import sys
import pickle as pkl

class NeuralNetwork():
    def __init__(self):
        self.w = {}
        self.b = {}
        self.h = {}
        self.n_layers = 1
        
    def add_input(self, sz):
        self.input_sz = sz
        
    def add_hidden(self, sz, act):
        self.h[self.n_layers] = {}
        self.h[self.n_layers]['sz'] = sz
        self.h[self.n_layers]['act'] = act
        if self.n_layers == 1:
            prev_sz = self.input_sz    
        else:
            prev_sz = self.h[self.n_layers-1]['sz']
        self.w[self.n_layers] = np.random.randn(prev_sz, sz)
        self.b[self.n_layers] = np.random.randn(sz, 1)
        self.n_layers += 1
        
    def add_output(self, sz):
        self.h[self.n_layers] = {}
        self.output_sz = sz
        prev_sz = self.h[self.n_layers-1]['sz']
        self.w[self.n_layers] = np.random.randn(prev_sz, sz)
        self.b[self.n_layers] = np.random.randn(sz, 1)
        
    def activation(self, act, z):
        if act == 'relu':
            return np.maximum(z, 0)
        elif act == 'linear':
            return z
        elif act == 'tanh':
            return np.tanh(z)
        elif act == 'sigmoid':
            return 1/(1+np.exp(-z))
        
    def softmax(self, z):
        return np.exp(z)/np.sum(np.exp(z))
        
    def activation_grad(self, act, a):
        if act == 'relu':
            return np.where(a>0, 1, 0)
        elif act == 'linear':
            return np.ones((a.shape[0], 1))
        elif act == 'tanh':
            return 1-a**2
        elif act == 'sigmoid':
            return a * (1-a)
    
    def crossentropy(self, y_true, y_hat):
        return -np.sum(y_true * np.log(y_hat))
            
    def sigma(self, a, w, b):
        return np.dot(w.T, a) + b
    
    def inf_nan_check(self, vector):
        if np.isnan(vector).any() or np.isinf(vector).any():
            return True 
    
    def accuracy(self, y_pred, y_true):
        return sum(y_pred == y_true)/len(y_pred)
        
    def validation_score(self):
        probs_val = list(map(lambda x: self.forward(x), self.inputs_val))
        y_pred_val = np.array([np.argmax(p) for p in probs_val])
        
        val_acc = self.accuracy(y_pred_val, self.y_label_val)
        val_loss = sum([self.crossentropy(y_true, y_hat) for y_true, y_hat in zip(self.targets_val, probs_val)])
        return val_acc, val_loss/len(y_pred_val)
    
    def forward(self, x):
        for i in range(self.n_layers):
            i+=1
            if i == 1:
                self.h[i]['z'] = self.sigma(x, self.w[i], self.b[i])
            else:
                self.h[i]['z'] = self.sigma(self.h[i-1]['a'], self.w[i], self.b[i])
            
            if i != self.n_layers:
                self.h[i]['a'] = self.activation(self.h[i]['act'], self.h[i]['z'])
            else:
                self.h[i]['a'] = self.softmax(self.h[i]['z'])
        
            if self.inf_nan_check(self.h[i]['a']):
                raise ValueError ("There is nan or inf value in activation output layer {}!!!".format(i))
        
        return self.h[i]['a']
    
    def backward(self, x, y_true, y_hat):
        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                self.h[i]['dL/dz'] = y_hat-y_true
                self.h[i]['dz/dw'] = np.tile(self.h[i-1]['a'], (1, self.output_sz))

            else:
                self.h[i]['dL/da'] = np.dot(self.w[i+1], self.h[i+1]['dL/dz'])
                self.h[i]['da/dz'] = self.activation_grad(self.h[i]['act'], self.h[i]['a'])
                self.h[i]['dL/dz'] = self.h[i]['dL/da'] * self.h[i]['da/dz']
                if i!=1:
                    self.h[i]['dz/dw'] = np.tile(self.h[i-1]['a'], (1, self.h[i]['sz']))
                else:
                    self.h[i]['dz/dw'] = np.tile(x, (1, self.h[i]['sz']))
                
            self.h[i]['dL/dw'] = self.h[i]['dL/dz'].T * self.h[i]['dz/dw']
            self.h[i]['dL/db'] = self.h[i]['dL/dz']
            
    def update_w(self):
        for i in range(self.n_layers):
            i+=1
            self.w[i] = self.w[i] - (self.lr * self.h[i]['dL/dw'])
            self.b[i] = self.b[i] - (self.lr * self.h[i]['dL/db'])
    
    def fit(self, inputs, targets, epochs, lr, validation_split=None, shuffle=False):
        self.lr = lr
        num_classes = targets[0].shape[0]
        assert inputs[0].shape[0] == self.input_sz
        assert targets[0].shape[0] == self.output_sz
        
        train_data = list(zip(inputs, targets))
        if shuffle:
            np.random.shuffle(train_data)
        
        if validation_split is not None:
            idx = int(validation_split * len(train_data))
            val_data = train_data[:idx]
            self.inputs_val, self.targets_val = list(zip(*val_data))
            self.y_label_val = np.array([np.argmax(t) for t in self.targets_val])
            
            train_data = train_data[idx:]
            
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            for idx, (x,y) in enumerate(train_data):
                y_hat = self.forward(x)
                loss = self.crossentropy(y, y_hat)/num_classes
                self.backward(x, y, y_hat)
                self.update_w()
                
                sys.stdout.write('\riter: {}/{}; loss: {:2f};'.format(
                    idx+1, len(train_data), loss))

            if validation_split is not None:
	            val_acc, val_loss = self.validation_score()
	            print("\nval_acc: {:2f}; val_loss: {:2f}".format(val_acc, val_loss/num_classes))
            print('-'*50)

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self.__dict__, f,2)