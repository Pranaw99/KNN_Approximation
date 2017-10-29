# KNN_Approximation
KNN_Approximation to improve the time complexity of the algorithm

# Problem statement
Usual KNN machine learning classification algorithm works in a way in which we need to calculate the k-nearest distance between all the training example
in the data points. If the dataset is small, we are good but if the dataset is huge which in practice do happen, you need to calculate the distance 
for all the datapoints which in terms of time complexity can be quite expensive operation.

To overcome this problem, I have implemented an approximate KNN- Algorithm, which may not be cent percent accurate but can provide good 
approximate result with quite a speed up in running time.

