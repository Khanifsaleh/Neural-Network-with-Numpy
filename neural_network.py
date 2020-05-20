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
            return np.ones((a.shape[0], a.shape[1]))
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
    
    def accuracy(self, pred, labels):
        return sum(pred == labels)/len(pred)
        
    def score(self, labels, y_true, y_hat_train=None, train=False, val=False):
        if train:
            probs = np.split(y_hat_train, self.batch_size, axis=1)
        elif val:
            probs = list(map(lambda x: self.forward(x), self.inputs_val))
        preds = np.array([np.argmax(p) for p in probs])
        acc = self.accuracy(preds, labels)
        loss = sum([self.crossentropy(y, y_hat) for y, y_hat in zip(y_true, probs)])
        return acc, loss/len(probs)
    
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
                self.h[i]['a'] = np.concatenate([self.softmax(self.h[i]['z'][:,j]).reshape(-1,1) 
                                                 for j in range(self.h[i]['z'].shape[1])], axis=1)
                
            if self.inf_nan_check(self.h[i]['a']):
                raise ValueError ("There is nan or inf value in activation output layer {}!!!".format(i))
        
        return self.h[i]['a']
    
    def backward(self, x, y_true, y_hat):
        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                self.h[i]['dL/dz'] = y_hat-y_true
                self.h[i]['dz/dw'] = [np.array([self.h[i-1]['a'][:,j],]*self.output_sz).T 
                                      for j in range(self.batch_size)]
                
            else:
                self.h[i]['dL/da'] = np.dot(self.w[i+1], self.h[i+1]['dL/dz'])
                self.h[i]['da/dz'] = self.activation_grad(self.h[i]['act'], self.h[i]['a'])
                self.h[i]['dL/dz'] = self.h[i]['dL/da'] * self.h[i]['da/dz']
                if i!=1:
                    self.h[i]['dz/dw'] = [np.array([self.h[i-1]['a'][:,j],]*self.h[i]['sz']).T 
                                          for j in range(self.batch_size)]
                else:
                    self.h[i]['dz/dw'] = [np.array([x[:,j],]*self.h[i]['sz']).T 
                                          for j in range(self.batch_size)]
            
            self.h[i]['dL/dw'] = sum([dl_dz * dz_dw for dl_dz, dz_dw 
                                      in zip(self.h[i]['dL/dz'].T, self.h[i]['dz/dw'])])/self.batch_size
            self.h[i]['dL/db'] = np.expand_dims(self.h[i]['dL/dz'].sum(axis=1)/self.batch_size, axis=1)
            
    def update_w(self):
        for i in range(self.n_layers):
            i+=1
            self.w[i] = self.w[i] - (self.lr * self.h[i]['dL/dw'])
            self.b[i] = self.b[i] - (self.lr * self.h[i]['dL/db'])

    def batching(self, data):
        num_batches = len(data)//self.batch_size
        data = data[:num_batches*self.batch_size]
        batch_data = []
        x, y = list(zip(*data))
        for i in range(0, len(data), self.batch_size):
            x_batch = np.concatenate(x[i:i+self.batch_size], axis=1)
            y_batch = np.concatenate(y[i:i+self.batch_size], axis=1)
            labels = [np.argmax(p) for p in y[i:i+self.batch_size]]
            batch_data.append((x_batch, y_batch, labels))
        return batch_data

    def prepare_data(self, inputs, targets, validation_split, shuffle):

        train_data = list(zip(inputs, targets))
        if shuffle:
            np.random.shuffle(train_data)
        
        if validation_split is not None:
            idx = int(validation_split * len(train_data))
            val_data = train_data[:idx]
            self.inputs_val, self.y_val = list(zip(*val_data))
            self.labels_val = np.array([np.argmax(t) for t in self.y_val])
            train_data = train_data[idx:]

        train_batch = self.batching(train_data)
        return train_batch
    
    def fit(self, inputs, targets, epochs, lr, batch_size, validation_split=None, shuffle=False):
        assert inputs[0].shape[0] == self.input_sz
        assert targets[0].shape[0] == self.output_sz
        num_classes = targets[0].shape[0]
        self.lr = lr
        self.batch_size = batch_size
        train_batch = self.prepare_data(inputs, targets, validation_split, shuffle)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            for idx, (x, y, labels) in enumerate(train_batch):
                y_hat = self.forward(x)
                self.backward(x, y, y_hat)
                self.update_w()
                train_acc, train_loss = self.score(labels, y, y_hat_train=y_hat, train=True)
                sys.stdout.write('\riter: {}/{}; train_acc: {:2f}; train_loss: {:2f}; '.format(
                    idx+1, len(train_batch), train_acc, train_loss/num_classes))

            if validation_split is not None:
                val_acc, val_loss = self.score(self.labels_val, self.y_val, val=True)
                print("\nval_acc: {:2f}; val_loss: {:2f}".format( val_acc, val_loss/num_classes))
            print('-'*75)

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self.__dict__, f,2)