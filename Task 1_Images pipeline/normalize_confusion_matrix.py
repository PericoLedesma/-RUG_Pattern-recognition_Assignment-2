import numpy as np

C = np.array(
[[58,  4,  1,  7,  0],
 [19, 15, 24,  0,  2],
 [23, 10, 27,  0,  0],
 [ 1,  0,  0, 53,  6],
 [ 5,  0,  1,  6, 68]])
cm = C
#C = C / C.astype(np.float).sum(axis=1)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
