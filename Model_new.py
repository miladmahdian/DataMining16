import numpy as np
from scipy import linalg,sparse
from numpy import dot

users = {};
songs ={};

print("Mapping each user to unique Index");
with open("kaggle_visible_evaluation_triplets.txt","r") as f:
        counter=0;
        for line in f:
            user,song,count=line.strip().split('\t')
            if user not in users:
                users[user] = counter;
                counter = counter+1;

print("Mapping each song to unique Index");
with open("kaggle_visible_evaluation_triplets.txt","r") as f:
        counter=0;
        for line in f:
            user,song,count=line.strip().split('\t')
            if song not in songs:
                songs[song] = counter;
                counter = counter+1;
I=[]
J=[]
V=[]

counter =0;
#a = np.zeros(shape=(len(users),len(songs)))
print("Creating Matrix");
with open("kaggle_visible_evaluation_triplets.txt","r") as f:
        for line in f:
            user,song,count=line.strip().split('\t')
            #a[ users[user] ][ songs[song]] = count
            I.append(users[user])
            J.append(songs[song])
            V.append(count)
            counter = counter +1;
            
A = sparse.coo_matrix((V,(I,J)), dtype=np.int8)

def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)

        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            print ('Iteration {}:'.format(i)),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print ('fit residual'), np.round(fit_residual, 4),
            print ('total residual'), np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y
    
nP, nQ = nmf(A, 2)

k = np.dot(nP,nQ)


#print (np.subtract(a,k))
