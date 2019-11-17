#----------------------------------------------------------------------------
# Q1: Modify "1_nn_pytorch.py" and save the performance.
# Plot the performance index with respect to epochs.
#----------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10         # Network output size
#----------------------------------------------------------------------------
p = torch.randn(Batch_size, R)
t = torch.randn(Batch_size, a_size, requires_grad=False)
#----------------------------------------------------------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)
#----------------------------------------------------------------------------
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------

learning_rate = 1e-4

index_list = []
loss_list = []
for index in range(500):

    a = model(p)
    loss = performance_index(a, t)
    #print(index, loss.item())
    index_list.append(index)
    loss_list.append(loss)

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad

plt.plot(index_list, loss_list)
plt.title('Performance index with respect to epochs')
plt.show()
plt.close()

#----------------------------------------------------------------------------
# Q2: Create a 4 layer network (for Q1) with the following sizes (1-100-50-20-1).
# Train the network and check the performance index.
Batch_size1 = 64     # Batch size
#----------------------------------------------------------------------------
p1 = torch.randn(Batch_size1, 1)
t1 = torch.randn(Batch_size1, 1, requires_grad=False)
#----------------------------------------------------------------------------
model1 = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1),
)
#----------------------------------------------------------------------------

learning_rate = 1e-4

index_list1 = []
loss_list1 = []
for index1 in range(500):

    a1 = model1(p1)
    loss1 = performance_index(a1, t1)
    #print(index1, loss1.item())
    index_list1.append(index1)
    loss_list1.append(loss1)

    model1.zero_grad()
    loss1.backward()

    for param1 in model1.parameters():
        param1.data -= learning_rate * param1.grad

#----------------------------------------------------------------------------
# Q3: Comparet the results of Q1 and Q2.
plt.plot(index_list, loss_list, label='Old')
plt.plot(index_list1, loss_list1, label='New')
plt.title('Performance index with respect to epochs')
plt.legend()
plt.show()
plt.close()

#----------------------------------------------------------------------------

# Q5:  Find all the parameters of the network and write a for loop to print
# the size and name of the parameters.
name_list = []
param_list = []
for name, param in model.named_parameters():
    if param.requires_grad:
        name_list.append(name)
        param_list.append(param.data.cpu().numpy())
        print(name, param.data)

name_list1 = []
param_list1 = []
for name1, param1 in model1.named_parameters():
    if param1.requires_grad:
        name_list1.append(name1)
        param_list1.append(param1.data.cpu().numpy())
        print(name1, param1.data)

#----------------------------------------------------------------------------'
# Q6: Save the gradient values for each weight in the csv file.
df = pd.DataFrame({'name': name_list, 'param': param_list})
print(df)
df.to_csv('model.csv')

df1 = pd.DataFrame({'name': name_list1, 'param': param_list1})
print(df1)
df1.to_csv('model1.csv')

#----------------------------------------------------------------------------'
# Q7: If you increase epochs for Q6 what changes do you see?
index_list = []
loss_list = []
for index in range(1000):

    a = model(p)
    loss = performance_index(a, t)
    #print(index, loss.item())
    index_list.append(index)
    loss_list.append(loss)

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad


name_list = []
param_list = []
for name, param in model.named_parameters():
    if param.requires_grad:
        name_list.append(name)
        param_list.append(param.data.cpu().numpy())
        print(name, param.data)

df_more = pd.DataFrame({'name': name_list, 'param': param_list})
print(df_more)
df_more.to_csv('model_more.csv')

name_list1 = []
param_list1 = []
for index1 in range(1000):

    a1 = model1(p1)
    loss1 = performance_index(a1, t1)
    #print(index1, loss1.item())
    index_list1.append(index1)
    loss_list1.append(loss1)

    model1.zero_grad()
    loss1.backward()

    for param1 in model1.parameters():
        param1.data -= learning_rate * param1.grad

name_list1 = []
param_list1 = []
for name1, param1 in model1.named_parameters():
    if param1.requires_grad:
        name_list1.append(name1)
        param_list1.append(param1.data.cpu().numpy())
        print(name1, param1.data)

df1_more = pd.DataFrame({'name': name_list1, 'param': param_list1})
print(df1_more)
df1_more.to_csv('model1_more.csv')

