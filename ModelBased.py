
import numpy
#import nimfa as nf

#users = {};
#songs ={};
#
#print("Mapping each user to unique Index");
#with open("kaggle_users.txt","r") as f:
#        count=0;
#        for line in f:
#            line = line.strip()
#            users[line] = count;
#            count = count+1;
#
#print("Mapping each song to unique Index");
#with open("kaggle_songs.txt","r") as f:
#        count=0;
#        for line in f:
#            song = line.rstrip('\n').split(' ')[0];
#	    id = int(line.rstrip('\n').split(' ')[1]);
#            songs[song] = id;
#            count = count+1;
#
#a = np.zeros(shape=(len(users)+1,len(songs)+1))
##9be82340a8b5ef32357fe5af957ccd54736ece95	SOHGGAH12A58A795BE	15
#
#print("Creating Matrix");
#with open("kaggle_visible_evaluation_triplets.txt","r") as f:
#        for line in f:
#            user,song,count=line.strip().split('\t')
#            a[ users[user] ][ songs[song]] = count
#
#print("Nimfa");            
##nmf = nf.Nmf(a, seed="nndsvd", rank=10, max_iter=1, update='euclidean',
##                objective='fro')
##nmf_fit = nmf()
#model = NMF(n_components=2, init='random', random_state=0)
#model.fit(a)

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

a = numpy.zeros(shape=(len(users),len(songs)))

P= numpy.ones(shape=(len(users),2))
Q = numpy.ones(shape=(len(songs),2))
print(a.shape);
print("Creating Matrix");
with open("kaggle_visible_evaluation_triplets.txt","r") as f:
        for line in f:
            user,song,count=line.strip().split('\t')
            a[ users[user] ][ songs[song]] = count

def matrix_factorization(R, P, Q, K, steps=10000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        print(step)
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T
    
nP, nQ = matrix_factorization(a, P, Q, 2)

nQt = numpy.transpose(nQ)
k = numpy.dot(nP,nQt)

print a.shape;
print k.shape;
            
print (numpy.subtract(a,k))
            
