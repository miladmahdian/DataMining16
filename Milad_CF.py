import time, math
import numpy as np


class Col_Filt:
    def __init__(self, n_items=1682, n_users=943, n_components=4, MAX_ITER=1000, lamd=0.05, eta=0.01, thresh = 0.01):
        self.n_components = n_components #number of lanent feature components
        self.MAX_ITER = MAX_ITER # number of iterations for grad descent
        self.lamd = lamd # lambda in the regularization
        self.eta = eta # step size in the grad descent
        self.n_users = n_users
        self.n_items = n_items
        self.thresh = thresh # change threshold to stop iterating



    def createMatrix(self,X):
        print ("Creating matrix")
        Rui = np.zeros((self.n_users, self.n_items,))
        for index in range(X.shape[0]):
            Rui[int(X[index,0])-1, int(X[index,1])-1] = X[index,2]
        return Rui

    def setRui(self,Rui):
        self.Rui = Rui


    def run(self,Rui):
        start = time.clock()
        self.Rui = Rui
        self.Pu = np.random.random((self.n_components, self.n_users))
        self.Qi = np.random.random((self.n_components, self.n_items))

        counter = 0
        cnt = 0
        e = 0.0
        ePrev = self.thresh + 1
        while abs(e - ePrev) >self.thresh and counter < self.MAX_ITER:
            ePrev = e
            e = 0
            for u in np.arange(self.n_users):
                for i in np.arange(self.n_items):
                    if Rui[u, i] > 0:
                        cnt += 1
                        eij = Rui[u, i] - self.Pu[:, u].T.dot(self.Qi[:, i])
                        self.Pu[:, u] += self.eta * (eij * self.Qi[:, i] - self.lamd * self.Pu[:, u])
                        self.Qi[:, i] += self.eta * (eij * self.Pu[:, u] - self.lamd * self.Qi[:, i])
                        e += (pow(eij,2)+self.lamd*(self.Pu[:, u].T.dot(self.Pu[:, u])+(self.Qi[:, i].T.dot(self.Qi[:, i]))))/self.n_items
                        #e += pow(Rui[u, i] - self.Pu[:, u].T.dot(self.Qi[:, i]),2)/self.n_items
            counter += 1
            print (counter)
            print (e)

        cti=time.clock()-start
        print ("Finished the gradient descent with time "+str(cti))
        print ("cnt: " + str(cnt))

    def eval_MAE (self,Rui_te):
        mae = 0.0
        counter = 0
        for u in np.arange(self.n_users):
                for i in np.arange(self.n_items):
                    if Rui_te[u, i] > 0:
                        counter += 1
                        mae += abs(Rui_te[u, i] - self.Pu[:, u].T.dot(self.Qi[:, i]))
                        #if counter % 100 == 0:
                         #   print("Rating %f is predicted by %f"%(Rui_te[u, i],self.Pu[:, u].T.dot(self.Qi[:, i])))

        return mae/counter

    def eval_RMSE (self,Rui_te):
        rmse = 0.0
        counter = 0
        for u in np.arange(self.n_users):
            for i in np.arange(self.n_items):
                if Rui_te[u, i] > 0:
                    counter += 1
                    rmse += math.pow(Rui_te[u, i] - self.Pu[:, u].T.dot(self.Qi[:, i]),2)

        return math.sqrt(rmse/counter)




    def save(self,stuff,size,ofile):
        f=open(ofile,"w")
        for i in range(size[0]):
            for j in range(size[1]):
                f.write(stuff[i,j]+"\t")
            f.write("\n")
        f.close()
        print ("Ok.")

