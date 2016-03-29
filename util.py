
import sys

import matplotlib.pyplot as plt
import numpy as np


tr = "train_triplets.txt"
a = []
with open(tr,"r") as read:
    for line in read:
        _,_,count=line.strip().split('\t')
        a.append(int(count))
        #if int(count) > max:
         #   max = int(count)

plt.hist(range(0, 9668))
plt.show()

#print("max is " + str(max)) 
            
#sys.stdout.flush()
