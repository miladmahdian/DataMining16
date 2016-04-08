import CF_model
import numpy as np
#import ModelBased


n_components=10

eta=0.01
lamd=0.05
count = 0
MAX_ITER=10
n_triplets = 0
users = {};
songs ={};

output = open('newfile.txt','w');
tr = "kaggle_visible_evaluation_triplets.txt";
print("Mapping each user and song to unique Index");
with open(tr,"r") as f:
        user_counter=0;
        song_counter=0;
        lines = '';
        for line in f:
            n_triplets+= 1 
            user,song,count=line.strip().split('\t')
            if song not in songs:
                songs[song] = song_counter;
                song_counter = song_counter+1;
                
            if user not in users:
                users[user] = user_counter;
                user_counter = user_counter+1;

            output.write( str(users[user]) +'\t'+ str(songs[song]) + '\t' + count)
                
        print("Created Tested and Training Files");
        output.close();
print("number of triplets is %d"%n_triplets) 
X = np.empty((n_triplets,3,))
count = 0
with open(tr, "r") as read:

    for line in read:
        user, item, rating = line.strip().split('\t')
        userId = int(user)
        itemId = int(item)
        rate = int(rating)
        X[count,:] = np.array([userId,itemId,rate])
        count += 1

mae = 0.0
rmse = 0.0
k = 5
binSize = n_triplets / k
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

    cf = CF_model.CFModel()
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
