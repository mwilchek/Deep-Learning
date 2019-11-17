#----------------------------------------------------------------------------'
# Q1: Try Adadelta optimizer for "3_nn_optim.py" file and use different values 
# for rho and eps.Do a little search and check and find out what are the effects 
# of each parameter.
#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable

#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10         # Network output size
#----------------------------------------------------------------------------
p = Variable(torch.randn(Batch_size, R))
t = Variable(torch.randn(Batch_size, a_size), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size ),
)
performance_index = torch.nn.MSELoss(size_average=False)
#----------------------------------------------------------------------------
learning_rate = 1e-4
#----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#----------------------------------------------------------------------------
for index in range(100):
    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.data.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#----------------------------------------------------------------------------'
# Q2:  Try SGD optimizer and use different values for momentum and weight decay. 
# Do a little search and check and find out what are the effects of each parameter.
optimizer2 = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
for index in range(100):
    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.data.item())
    optimizer2.zero_grad()
    loss.backward()
    optimizer.step()

#----------------------------------------------------------------------------'
# Q3:  Try Adam optimizer and find the effect of beta.
# Do a little search and check and find out what are the effects of each parameter.
optimizer3 = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, weight_decay=0,  lr=learning_rate)
for index in range(100):
    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.data.item())
    optimizer3.zero_grad()
    loss.backward()
    optimizer.step()
#----------------------------------------------------------------------------'
# Note: I suggest you to the the followings but you do not need to submit it.

#  List all of the optimizers and check all of them in the following format.

#  1- Open the documentation and check the arguments (how many parameters it needs)
#  2- Read the theory behind the optimizer and compare it with your optimization knowledge.
#  3- Write a summary half a page in your own words
#  4- Test it practically with the code and validate your findings.



