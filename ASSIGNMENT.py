import numpy as np
import math
from scipy import linalg,sparse
from numpy import dot
#Initial Frame
#Matrix Factorization
def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
        Decompose X to A*Y
        """
    eps = 0.001
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
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
            #print 'Iteration {}:'.format(i),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est
            
            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            #print 'fit residual', np.round(fit_residual, 4),
            #print 'total residual', np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

return A, Y

lines = []
with open("u.data","r") as f:
    for line in f:
        lines.append(line);

folds =5;
foldSize = 100000/5;
for i in range(0, folds):
    start = i*foldSize;
    end = start+ foldSize -1
    train = [];
    test = [];
    I=[]
    J=[]
    V=[]
    for j in range( 0 , 100000):
        user,movie,rating,time=lines[j].strip().split('\t')
        # testing
        if (j >=start and j<= end):
            rating =0;
            test.append(lines[j])
        I.append(int(user))
        J.append(int(movie))
        V.append(int(rating))
    
    A = sparse.coo_matrix((V,(I,J)), dtype=np.int8)
    print len(V)
    nP, nQ = nmf(A, 2)
    k = np.dot(nP,nQ)
    absErr = 0;
    absErrs = 0
    for o in range(0,len(test)):
        user,movie,rating,time=test[o].strip().split('\t')
        ratingsCalc = k[int(user)][int(movie)]
        diff = abs(ratingsCalc - int(rating))
        diffs = math.pow(diff,2)
        absErr += diff
        absErrs += diffs
    MAE = absErr / len(test)
    RMAE = math.pow(absErrs / len(test), 0.5)
    print MAE
    print RMAE





