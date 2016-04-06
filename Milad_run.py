import Milad_CF
import numpy as np
#import ModelBased

tr = "u.data"
n_ratings = 100000
n_items=1682
n_users=943
n_components=10
X = np.empty((n_ratings,3,))
R = np.empty((n_users, n_items,))
eta=0.01
lamd=0.05
count = 0
MAX_ITER=10

with open(tr, "r") as read:

    for line in read:
        user, item, rating, _ = line.strip().split('\t')
        userId = int(user)
        itemId = int(item)
        rate = int(rating)
        X[count,:] = np.array([userId,itemId,rate])
        count += 1

mae = 0.0
rmse = 0.0
k = 5
binSize = n_ratings / k
for  i in range(k):
    if i == 0 :
        trainSet = X[binSize:,:]
        testSet = X[0:binSize,:]
    elif i == k - 1 :
        trainSet = X[0:(k - 1) * binSize,:]
        testSet = X[(k - 1) * binSize:,:]
    else:
        testSet = X[i * binSize: (i + 1) * binSize,:]
        tr1 = X[:i * binSize,:]
        tr2 = X[(i + 1) * binSize:,:]
        trainSet = np.vstack([tr1,tr2])
    #P = np.random.random(( n_users,10))
    #Q = np.random.random(( n_items,10))
    #nP, nQ = ModelBased.matrix_factorization(trainSet, P, Q, 10)
    #temp = 0.0
    #for u in np.arange(n_users):
     #   for i in np.arange(n_items):
            #if testSet[u, i] > 0:
      #          temp += abs(testSet[u, i] - nP[:, u].T.dot(nQ[:, i]))

    cf = Milad_CF.Col_Filt()
    Rui_tr = cf.createMatrix(trainSet)
    Rui_te = cf.createMatrix(testSet)
    cf.run(Rui_tr)
    temp = cf.eval_MAE(Rui_te)
    print("Round %d: MAE = %f"%(i,temp))

    mae += temp

    temp = cf.eval_RMSE(Rui_te)
    print("Round %d: RMSE = %f"%(i,temp))
    rmse += temp

print("The average MAE = %f"%(mae/k))
print("The average RMSE = %f"%(rmse/k))
