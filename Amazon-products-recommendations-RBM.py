# Amazon Product Recommendations using Restricted Boltzmann Machines (RBM)

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Preparing the dataset
df = pd.read_csv('ratings_Electronics.csv', names=['User_id','Product_id','Rating','Timestamp'])
data = np.array(df)
# Since the dataset is very huge (contains more than 7 million rows and too many products and ratings), 
# here we are taking a subset of that for our current study
train = data[7804481:7814481,:]

df_train = pd.DataFrame(train, columns=['User_id','Product_id','Rating','Timestamp'])


Number_of_users = df_train['User_id'].nunique()
unique_user_id = df_train['User_id'].unique()
unique_user_id_list = unique_user_id.tolist()
Number_of_products = df_train['Product_id'].nunique()
unique_product_id = df_train['Product_id'].unique()
unique_product_id_list = unique_product_id.tolist()

dict = {}
for i,product_id in enumerate(unique_product_id_list):
    dict[i] = product_id

inv_dict = {v: k for k, v in dict.items()}

# Adding extra column (product number) to the dataframe
df_train['Product_no'] = df_train['Product_id'].map(inv_dict)

training_set =  np.array(df_train)


# Converting the data into an array with users in lines and products in columns
def convert(data):
    new_data = []
    for id_users in unique_user_id_list:
        num_products = data[:,4][data[:,0] == id_users]
        num_products = num_products.reshape(-1,1)
        id_ratings = data[:,2][data[:,0] == id_users]
        id_ratings = id_ratings.reshape(-1,1)
        num_products = num_products.astype('int')
        id_ratings = id_ratings.astype('int')
        
        ratings = np.zeros(Number_of_products)
        ratings[num_products] = id_ratings
        
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)

# Converting the ratings into binary ratings 1 (Liked the product) or 0 (Not Liked the product)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

# Creating the architecture of the Neural Network for RBM
class RBM():
    def __init__(self, nv, nh, lr):       # nv-> no. of visible nodes, nh-> no. pf hidden nodes
        self.W = torch.randn(nh, nv)  # initializing the wts of the probabilities of the visible nodes given hidden nodes)
        self.a = torch.randn(1, nh)   # bias for the probability of the hidden nodes given visible nodes (1 -> extra dimension corresponds to the batch, nh corresponds to the bias) 
        self.b = torch.randn(1, nv)   # bias for the probability of the visible nodes given hidden nodes (1 -> extra dimension corresponds to the batch, nv corresponds to the bias) 
        self.lr = lr
    def sample_h(self, x):   # x-> visible neurons in the probabilities p(h|v) (sigmoid activation fn)
        wx = torch.mm(x, self.W.t())  # tensors multiplication of x and weights 
        activation = wx + self.a.expand_as(wx) # activation-> wx + a (bias) , expanding the dimension of a to match wx dimension
        p_h_given_v = torch.sigmoid(activation)  # applying sigmoid activation fn to calculate probability
        return p_h_given_v, torch.bernoulli(p_h_given_v)   # return p_h_given_v, return some samples (Bernoulli's samples) of the hidden neurons given this probability
    def sample_v(self, y):  # y-> hidden neurons in the probabilities p(v|h) (sigmoid activation fn)
        wy = torch.mm(y, self.W)   # tensors multiplication of y and weights
        activation = wy + self.b.expand_as(wy) # activation-> wy + b (bias) , expanding the dimension of a to match wy dimension
        p_v_given_h = torch.sigmoid(activation)  # applying sigmoid activation fn to calculate probability
        return p_v_given_h, torch.bernoulli(p_v_given_h)  # return p_v_given_h, return some samples (Bernoulli's samples) of the visible neurons given this probability
    def train(self, v0, vk, ph0, phk): # Contrastive Divergence (approximation of the log likelyhood gradient). 
                                       # Gibbs Sampling (creating Gibbs chain in k steps i.e by sampling k times the 
                                       # visible nodes and the hidden nodes).  v0->visible node, vk-> visible node obtained after k sampling
                                       # ph0-> probability of hidden node at first iteration given v0
                                       # phk-> probability of hidden node after k sampling given vk
        self.W += ((torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t())*lr  # updating tensor of wts -> W = W + (v0.ph0 - vk.phk)*learning-rate
        self.b += torch.sum((v0 - vk), 0)  # updating bias for visible node -> b = b + (v0 - vk)
        self.a += torch.sum((ph0 - phk), 0)  # updating bias for hidden node -> a = a + (ph0 - phk)

nv = Number_of_products
nh = 100
lr = 0.1
batch_size = 100
rbm = RBM(nv, nh, lr)


# Training the RBM
print('Training started.............')
nb_epoch = 30
loss_data = np.zeros((nb_epoch,2)) 

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, Number_of_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]  # we are keeping original -ve(-1) ratings
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE (Root Mean Square Error) Loss
        s += 1.
    
    loss_data[epoch-1,0] = epoch
    loss_data[epoch-1,1] = (train_loss/s)
    print('epoch: '+str(epoch)+' Training loss(RMSE): '+str(train_loss/s))

# Plotting of Training Loss
import matplotlib.pyplot as plt
plt.plot(loss_data[:,0], loss_data[:,1])
plt.xlabel('epoch')
plt.ylabel('Training Loss (RMSE)')
plt.show()


# Original data
training_set_array_orig = training_set.numpy()


# Function for predictions/product recommendations 
def rbm_pred(x):
    if len(x[x>=0]) > 0:
        _,h = rbm.sample_h(x)
        _,v = rbm.sample_v(h)
    return v

# Single User's predictions (Product Recommendations by RBM from re-constructed Inputs for the products that he/she did not buy/rate)
    
single_user_number = np.random.randint(0,20)

single_user_predictions = rbm_pred(training_set[single_user_number:single_user_number+1])

single_user_predictions = single_user_predictions.numpy()

single_user_orig_inputs = training_set_array_orig[single_user_number:single_user_number+1]

single_user_predictions_df = pd.DataFrame(single_user_predictions, columns=[unique_product_id_list])


file_name = 'RBM_product_recommendations_for_single_user_id_' + unique_user_id_list[single_user_number] + '.csv'
single_user_predictions_df.to_csv(file_name, index=False)

single_user_orig_inputs_df = pd.DataFrame(single_user_orig_inputs, columns=[unique_product_id_list])


file_name1 = 'Original_product_ratings_for_single_user_id_' + unique_user_id_list[single_user_number] + '.csv'

single_user_orig_inputs_df.to_csv(file_name1, index=False)
















