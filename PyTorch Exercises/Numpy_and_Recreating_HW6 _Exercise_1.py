# --------------------------------------------------------------------------
# Q1:  Rewrite the "2_tensor_pytorch.py" file in Neural Network Design Book,
# Chapter 11 notation (Check the Summary Page) and find out
# what are the differences. Explain your findings.
print("Based on the re-design, the largest differences creates less loss. The added backpropagation and revised "
      "learning rates and weights creates a more accurate model with less loss.")


# --------------------------------------------------------------------------
# Q2: Use the time package of python and test it on "1_Numpy.py" code and save the
# running time. Change the dtype to torch tensor save the running time as well.
# Compare the timing results. Explain your findings.
print("Running time of original 1_Numpy.py is 0.7509598731994629 seconds. "
      "After I change the dtype to torch tensor, the running time reduces to 0.30742740631103516 seconds. " 
      "Torch tensor is more efficient than numpy.")


# --------------------------------------------------------------------------
# Q3: Keep the data size same and change the number of epochs for Q2.
# Comapre the timing results. Explain your findings.
print("At epochs of 50:"
      "Numpy version has a running time of 0.08095812797546387 seconds."
      "Torch tensor version has a running time of  0.12669682502746582 seconds."
      
      "At epochs of 3000:"
      "Numpy version has a running time of 4.459993124008179 seconds."
      "Torch tensor version has a running time of 1.364422082901001 seconds." 
      
      "On the small epochs, the performance of numpy is better than torch tensor." 
      "On the large epochs, torch tensor's efficiency is way better than numpy.")


# --------------------------------------------------------------------------
# Q4: Increase the data size and keep the number of epochs for Q2 (Hints: Big number for epochs).
# Comapre the timing results. Explain your findings.
print("At epochs of 3000:"
      
      "Input size is 50:"
      "Numpy version has a running time of  0.7389566898345947 seconds."
      "Torch tensor version has a running time of 0.9114272594451904 seconds." 
      
      "Input size is 3000:"
      "Numpy version has a running time of 11.019431352615356 seconds."
      "Torch tensor version has a running time of 1.9880273342132568 seconds." 
      
      "On a big number for epochs: "
      "On the small data size, the performance of numpy is better than torch tensor." 
      "On the large data size, torch tensor's efficiency is way better than numpy." 
      "The performance of numpy is more sensitive to data size.")


# ----------------------------------------------------------------------------'
# Q5: Keep the data size big and keep the number of epochs big. Change the dtype to
# torch tensor cuda and compare it with numpy.
# Comapre the timing results. Explain your findings.

print("At epochs of 3000:"
      "Input size is 3000:"
      "Numpy version has a running time of 10.503144264221191 seconds."
      "Torch tensor version has a running time of 1.2609915733337402 seconds."

      "On a big number for epochs and data size: "
      "The performance of torch tensor is way better than numpy."
      "The performance of torch tensor cuda is also better than torch tensor cpu.")
