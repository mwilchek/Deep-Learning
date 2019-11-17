#----------------------------------------------------------------------------'
# Q1: Modify "3_nn_optim.py" inorder to find our the cuda capabilities of the system.
#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
p = Variable(torch.randn(Batch_size, R).cuda())
t = Variable(torch.randn(Batch_size, a_size).cuda(), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)

model.cuda()
performance_index = torch.nn.MSELoss(size_average=False)
#----------------------------------------------------------------------------
learning_rate = 1e-4
#----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#----------------------------------------------------------------------------
for index in range(500):
    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.data.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#----------------------------------------------------------------------------'
# Q2: Train a 2 layer mlp network using pytorch to approximate the the following function 
# t= sin(p)  ; -3< p < 3
p2 = np.linspace(-3, 3, 100)  # (100,)
t2 = np.sin(p2)  # (100,)

# Converts to Tensors and reshapes to suitable shape (n_examples, 1)
# requires_grad=True on the input so that the gradients are computed when calling loss.backward()
# i.e, so that all the operations performed on p and on their outputs are made part of the Computational Graph
p2 = torch.Tensor(p2).reshape(-1, 1)
p2.requires_grad = True
t2 = torch.Tensor(t2).reshape(-1, 1)
model2 = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)
criterion2 = nn.MSELoss()
#----------------------------------------------------------------------------
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
for epoch in range(100):
    # Sets the gradients stored on the .grad attribute of each parameter from the previous iteration to 0
    optimizer2.zero_grad()  # It is good practice to do it right before going forward on any model
    # Goes forward (doing full batch here), notice we don't need to do model.forward(p)
    t_pred2 = model2(p2)
    # Computes the mse
    loss2 = criterion2(t2, t_pred2)
    # Goes backwards (computes all the gradients of the mse w.r.t the parameters
    # starting from the output layer all the way to the input layer)
    loss2.backward()
    # Updates all the parameters using the gradients which were just computed
    optimizer2.step()
    # Checks the training process
    if epoch % 1 == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss2.item()))

#----------------------------------------------------------------------------'
# Q3: Redo the Q2 with 6 layer network.
p3 = np.linspace(-3, 3, 100)  # (100,)
t3 = np.sin(p3)  # (100,)
# Converts to Tensors and reshapes to suitable shape (n_examples, 1)
# requires_grad=True on the input so that the gradients are computed when calling loss.backward()
# i.e, so that all the operations performed on p and on their outputs are made part of the Computational Graph
p3 = torch.Tensor(p3).reshape(-1, 1)
p2.requires_grad = True
t3 = torch.Tensor(t3).reshape(-1, 1)
model3 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 7),
    torch.nn.ReLU(),
    torch.nn.Linear(7, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2),
    torch.nn.ReLU(),
    torch.nn.Linear(2, 1)
)
criterion3 = nn.MSELoss()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
for epoch in range(100):
    # Sets the gradients stored on the .grad attribute of each parameter from the previous iteration to 0
    optimizer3.zero_grad()  # It is good practice to do it right before going forward on any model
    # Goes forward (doing full batch here), notice we don't need to do model.forward(p)
    t_pred3 = model3(p3)
    # Computes the mse
    loss3 = criterion3(t3, t_pred3)
    # Goes backwards (computes all the gradients of the mse w.r.t the parameters
    # starting from the output layer all the way to the input layer)
    loss3.backward()
    # Updates all the parameters using the gradients which were just computed
    optimizer3.step()
    # Checks the training process
    if epoch % 1 == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss3.item()))