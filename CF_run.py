import CF_model, math
import numpy as np
#import ModelBased


components=5

eta=0.01
lamd=0.05
count = 0
MAX_ITER=200

n_triplets = 7858609#48373586#1450933
user_counter=26386#1019318 #110000;
song_counter=297053#384546 #163206;

print("number of triplets is %d, users %d and songs %d"%(n_triplets,user_counter,song_counter))
X = np.empty((n_triplets,3,))
count = 0
with open("train_triplets_concise.txt","r") as f:
        for line in f:
            user, item, rating = line.strip().split('\t')
            userId = int(user)
            itemId = int(item)
            rate = 1+math.log(int(rating))
            X[count,:] = np.array([userId,itemId,rate])
            count += 1

mae = 0.0
rmse = 0.0
k = 5
binSize = n_triplets / k
indexes = range(n_triplets)
np.random.shuffle(indexes)
Xs = X[indexes,:]
del X
for  i in range(k):
    if i == 0 :
        trainSet = Xs[binSize:,:]
        testSet = Xs[0:binSize,:]
    elif i == k - 1 :
        trainSet = Xs[0:(k - 1) * binSize,:]
        testSet = Xs[(k - 1) * binSize:,:]
    else:
        testSet = Xs[i * binSize: (i + 1) * binSize,:]
        tr1 = Xs[:i * binSize,:]
        tr2 = Xs[(i + 1) * binSize:,:]
        trainSet = np.vstack([tr1,tr2])
    #P = np.random.random(( n_users,10))
    #Q = np.random.random(( n_items,10))
    #nP, nQ = ModelBased.matrix_factorization(trainSet, P, Q, 10)
    #temp = 0.0
    #for u in np.arange(n_users):
     #   for i in np.arange(n_items):
            #if testSet[u, i] > 0:
      #          temp += abs(testSet[u, i] - nP[:, u].T.dot(nQ[:, i]))

    cf = CF_model.CFModel(n_items=song_counter, n_users=user_counter, n_components=components)
    Rui_tr = cf.createMap(trainSet)
    Rui_te = cf.createMap(testSet)
    cf.run(Rui_tr)
    temp = cf.eval_MAE(Rui_te)
    print("Round %d: MAE = %f"%(i,temp))

    mae += temp

    temp = cf.eval_RMSE(Rui_te)
    print("Round %d: RMSE = %f"%(i,temp))
    rmse += temp

print("The average MAE = %f"%(mae/k))
print("The average RMSE = %f"%(rmse/k))
