#----------------------------------------------------------------------------
# Q1: Modify  the "1_Numpy.py" file and chane the dtype to torch float tensor.
# Save the value of the performance and plot the followings:
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
start_time1 = time.time()
#----------------------------------------------------------------------------
dtype = torch.float
#device = torch.device("cuda:0")
device = torch.device("cpu")

# ----------------------------------------------------------------------------
p = torch.randn(Batch_size, R, device=device, dtype=dtype)
t = torch.randn(Batch_size, a, device=device, dtype=dtype)
# ----------------------------------------------------------------------------
# Randomly initialize weights
w1 = torch.randn(R, S, device=device, dtype=dtype)
w2 = torch.randn(S, a, device=device, dtype=dtype)

learning_rate = 1e-6

# ----------------------------------------------------------------------------
index_list = []
loss_list = []
w1_list = []
w2_list = []
for index in range(100):

    h = p.mm(w1)
    h_relu = h.clamp(min=0)
    a_net = h_relu.mm(w2)

    loss = (a_net - t).pow(2).sum()
    #print(index, loss)
    index_list.append(index)
    loss_list.append(loss)

    grad_y_pred = 2.0 * (a_net - t)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = p.t().mm(grad_h)

    w1_list.append(grad_w1.mean())
    w2_list.append(grad_w2.mean())

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2


print("--- %s seconds ---" % (time.time() - start_time1))

#----------------------------------------------------------------------------
# i. Performance index with respect to epochs.
plt.plot(index_list, loss_list)
plt.title('Performance index with respect to epochs')
plt.show()
plt.close()
#----------------------------------------------------------------------------
# ii. w1 grad
plt.plot(index_list, w1_list)
plt.title('Performance w1 grad with respect to epochs')
plt.show()
plt.close()
#----------------------------------------------------------------------------
# ii. w2 grad
plt.plot(index_list, w2_list)
plt.title('Performance w2 grad with respect to epochs')
plt.show()
plt.close()
#----------------------------------------------------------------------------
# iii. Check your results. Explain each of your plots.
print("Since the loss converges to zero at around 20 echoes, "
      "I set the max echo to 100 for best visualization. " 
      "Also w1 grad and w2 grad with respect to epochs converge to zero when loss is minimized. ")