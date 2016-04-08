import time, math
import numpy as np


class CFModel:
    def __init__(self, n_items, n_users, n_components, MAX_ITER=200, lamd=0.05, eta=0.005, thresh = 0.01):
        self.n_components = n_components #number of lanent feature components
        self.MAX_ITER = MAX_ITER # number of iterations for grad descent
        self.lamd = lamd # lambda in the regularization
        self.eta = eta # step size in the grad descent
        self.n_users = n_users
        self.n_items = n_items
        self.thresh = thresh # change threshold to stop iterating


    def createMap(self,X):
        print ("Creating map")
        #Rui = np.zeros((self.n_users, self.n_items,))
        Rui = dict()
        self.n_triplets = X.shape[0]
        max = -1
        min = 10000000000
        for index in range(self.n_triplets):
            Rui[(int(X[index,0]), int(X[index,1]))] = X[index,2]
            if max <X[index,2]:
                max = X[index,2]
            if min > X[index,2]:
                min = X[index,2]
        print("max is %f and min is %f"%(max,min))
        return Rui


    def setRui(self,Rui):
        self.Rui = Rui

    def run(self,Rui):
        start=time.clock()
        self.Rui = Rui
        #        components = [1.0/self.n_components]* self.n_components
        #        self.Pu = dict.fromkeys(range(self.n_users),components)

        self.Pu = np.random.random((self.n_components, self.n_users))
        self.Qi = np.random.random((self.n_components, self.n_items))
        counter = 0
        e = 0.0
        ePrev = self.thresh + 1
        while abs(e - ePrev) >self.thresh and counter < self.MAX_ITER:
            ePrev = e
            e = 0
            cnt = 0
            #for u in np.arange(self.n_users):
            #    for i in np.arange(self.n_items):
            for k, rui in Rui.iteritems():
                    #rui = Rui[(u, i)]
                    if rui > 0:
                        cnt += 1
                        #print (rui)
                        #print(self.Pu[:, k[0]])
                        #print(self.Qi[:, k[1]])
                        eij = rui - self.Pu[:, k[0]].T.dot(self.Qi[:, k[1]])
                        self.Pu[:, k[0]] += self.eta * (eij * self.Qi[:, k[1]] - self.lamd * self.Pu[:, k[0]])
                        self.Qi[:, k[1]] += self.eta * (eij * self.Pu[:, k[0]] - self.lamd * self.Qi[:, k[1]])
                        e += (pow(eij,2)+self.lamd*(self.Pu[:, k[0]].T.dot(self.Pu[:, k[0]])+(self.Qi[:, k[1]].T.dot(self.Qi[:, k[1]]))))/self.n_triplets
                        #e += pow(Rui[u, i] - self.Pu[:, u].T.dot(self.Qi[:, i]),2)/self.n_items
            counter += 1
            print (counter)
            print (e)
        cti=time.clock()-start
        print ("Finished the gradient descent with time "+str(cti))
        print("cnt : "+str(cnt))
    

        
    def eval_MAE (self,Rui_te):
        mae = 0.0
        counter = 0
        #for u in np.arange(self.n_users):
        #        for i in np.arange(self.n_items):
        for k, rui in Rui_te.iteritems():
                    #rui = Rui_te[(u, i)]
                    if rui > 0:
                        counter += 1
                        mae += abs(rui - self.Pu[:, k[0]].T.dot(self.Qi[:, k[1]]))/1e6
                        #if counter % 100 == 0:
                         #   print("Rating %f is predicted by %f"%(Rui_te[u, i],self.Pu[:, u].T.dot(self.Qi[:, i])))

        return mae*(1e6/counter)

    def eval_RMSE (self,Rui_te):
        rmse = 0.0
        counter = 0
        #for u in np.arange(self.n_users):
        #    for i in np.arange(self.n_items):
        for k, rui in Rui_te.iteritems():
                #rui = Rui_te[(u, i)]
                if rui > 0:
                    counter += 1
                    rmse += math.pow((rui - self.Pu[:, k[0]].T.dot(self.Qi[:, k[1]])),2)

        return math.sqrt(rmse/counter)




    def save(self,stuff,size,ofile):
        f=open(ofile,"w")
        for i in range(size[0]):
            for j in range(size[1]):
                f.write(stuff[i,j]+"\t")
            f.write("\n")
        f.close()
        print ("Ok.")

