import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Give name to features and target
headers =  ['age', 'sex','chest_pain','resting_blood_pressure',
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
        'num_of_major_vessels','thal', 'heart_disease']
heart_df = pd.read_csv('heart.dat', sep=' ', names=headers)

# Drop target from data and replace target class with 0 and 1
# 0 means "have heart disease" and 1 means "do not have heart disease"
X = heart_df.drop(columns=['heart_disease'])
heart_df['heart_disease'] = heart_df['heart_disease'].replace(1,0)
heart_df['heart_disease'] = heart_df['heart_disease'].replace(2,1)
y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size = 0.2, random_state = 2)

# Let us standardize the dataset
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Define the neural network class
class NeuralNetwork():

    # This is a two layer neural network with 13 input nodes, 8 nodes in hidden layer
    # and 1 output node

    def __init__(self, layers = [13,8,1], learning_rate = 0.001, iterations = 100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None

    def init_weights(self):

        # We initialize the weights from a random normal distribution
        np.random.seed(1) # Seed the random number generator
        self.params['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2],)


    # Let us add activation functions to our class

    def relu(self,Z):
        return np.maximum(0,Z)

    def sigmoid(self,Z):
        return 1.0/(1+np.exp(-Z))

    # We will also need a loss function.
    # For a classification problem we use the cross-entropy loss

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat),y)+np.multiply((1-y),np.log(1-yhat))))
        return loss

    # Forward propagation
    def forward(self):

        Z1 = self.X.dot(self.params['W1'])+self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2'])+self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y,yhat)

        # Save the calculated params to be used for backpropagation
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat, loss


    # Backward propagation
    def backward(self, yhat):

        # Abbreviation: Differentiate loss with respect to -> dl_wrt_...

        # Derivative of Relu
        def d_Relu(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x

        dl_wrt_yhat = -(np.divide(self.y,yhat)-np.divide((1-self.y),(1-yhat)))
        dl_wrt_sig = yhat*(1-yhat)
        dl_wrt_Z2 = dl_wrt_yhat*dl_wrt_sig

        dl_wrt_A1 = dl_wrt_Z2.dot(self.params['W2'].T)
        dl_wrt_W2 = self.params['A1'].T.dot(dl_wrt_Z2)
        dl_wrt_b2 = np.sum(dl_wrt_Z2, axis = 0)

        dl_wrt_Z1 = dl_wrt_A1*d_Relu(self.params['Z1'])
        dl_wrt_W1 = self.X.T.dot(dl_wrt_Z1)
        dl_wrt_b1 = np.sum(dl_wrt_Z1, axis = 0)

        # Update weights and biases
        self.params['W1'] = self.params['W1']-self.learning_rate*dl_wrt_W1
        self.params['b1'] = self.params['b1']-self.learning_rate*dl_wrt_b1
        self.params['W2'] = self.params['W2']-self.learning_rate*dl_wrt_W2
        self.params['b2'] = self.params['b2']-self.learning_rate*dl_wrt_b2


    def fit(self, X, y):

        self.X = X
        self.y = y
        self.init_weights() # initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward()
            self.backward(yhat)
            self.loss.append(loss)

    def predict(self, X):

        Z1 = self.X.dot(self.params['W1'])+self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2'])+self.params['b2']
        prediction = self.sigmoid(Z2)
        return np.round(prediction)

    def accuracy(self, y, yhat):

        acc = int(sum(y==yhat)/len(y)*100)
        return acc

    def plot_loss(self):

        plt.plot(self.loss)
        plt.xlabel("Iterations")
        plt.ylabel("logloss")
        plt.title("Loss curve")
        plt.savefig('Logloss.pdf')
        plt.show()

# Create the neural network model
nn = NeuralNetwork()
nn.fit(X_train,y_train)
nn.plot_loss()











