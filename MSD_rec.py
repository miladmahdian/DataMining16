import os,sys,random,math,time
import MSD_util

def fl():
    sys.stdout.flush()

#l_rec: list of recommended songs
#u2s: mapping users to songs
#tau: 500
def AP(l_rec, sMu, tau):

    np=len(sMu)
    #print "np:", np
    nc=0.0
    mapr_user=0.0
    for j,s in enumerate(l_rec):
        if j>=tau:
            break
        if s in sMu:
        #print "s in sMu"
            nc+=1.0
            mapr_user+=nc/(j+1)
    mapr_user/=min(np,tau)
    return mapr_user

#l_users: list of users
#l_rec_songs: list of lists, recommended songs for users
#u2s: mapping users to songs
#tau: 500
def mAP(l_users, l_rec_songs, u2s, tau):
    mapr=0
    n_users=len(l_users)
    for i,l_rec in enumerate(l_rec_songs):
        if not l_users[i] in u2s:
            continue
        mapr+=AP(l_rec,u2s[l_users[i]], tau)
    return mapr/n_users

###
### PREDICTORS
###

class Pred:
    '''Implement generic predictor'''        
    
    def __init__(self):
        pass

    def Score(self,user_songs,  all_songs):
        return {}

class PredSI(Pred):
    '''Implement song-similarity based predictor'''

    def __init__(self, _s2u_tr, _A=0, _Q=1):
        Pred.__init__(self)
        self.s2u_tr = _s2u_tr
        self.Q = _Q
        self.A = _A
# prints the parameters A and Q
    def printati(self):
        print ("PredSI(A=%f,Q=%f)"%(self.A,self.Q),)
     # returns W_{i,j} in eq. (2)   
    def Match(self,s,u_song):
        l1=len(self.s2u_tr[s])
        l2=len(self.s2u_tr[u_song])
        up = float(len(self.s2u_tr[s]&self.s2u_tr[u_song])) # |I(u)^I(v)|
        if up>0:
            dn = math.pow(l1,self.A)*math.pow(l2,(1.0-self.A))#|I(u)|^a * |I(v)|^(1-a)
            return up/dn
        return 0.0
#user_songs: the set of songs which a particular user has listened to
# returns a dict (song from all_songs, score); the score is calculated as follows:
#it calculates W_{u,v} in eq. (2) where u is a song among user_songs and v is a song among all_songs
# it somes up W_{u,v}^Q over all the user_songs(u in user_songs). This is the  score for a song v from the training triplets
    def Score(self,user_songs,  all_songs):
        s_scores={}
        for s in all_songs:
            s_scores[s]=0.0
            if not (s in self.s2u_tr):
                continue  # i don't think this ever happens, to make sure we can put a print here. because both are originated from the training triplets
            for u_song in user_songs:
                if not (u_song in self.s2u_tr):
                    continue # if u_song is not in the training triplet: it is a new song
                s_match=self.Match(s,u_song)
                s_scores[s]+=math.pow(s_match,self.Q)
        return s_scores

class PredSIc(PredSI):
    '''Implement calibrated song-similarity based predictor''' 

    def __init__(self, _s2u_tr, _A=0, _Q=1, f_hsongs=""):
        PredSI.__init__(self, _s2u_tr, _A, _Q)
        self.hsongs={}
        with open(f_hsongs,"r") as f:
            for line in f:
                s,v = line.strip().split()
                self.hsongs[s]=float(v)
        self.THETA = 0.5

    def select_theta(self,h):
        return self.THETA
        
    def calibrate(self, sco, song):
        h = self.hsongs[song]
        theta = self.select_theta(h)
        prob=sco
        if sco<h:
            prob = theta*sco/h
        elif sco>h:
            prob = theta+(1.0-theta)*(sco-h)/(1.0-h)
        return prob

    def Score(self, user_songs,  all_songs):
        np = len(user_songs)
        s_scores={}
        for s in all_songs:
            s_scores[s]=0.0
            if not (s in self.s2u_tr):
                continue
            for u_song in user_songs:
                if not (u_song in self.s2u_tr):
                    continue
                s_match=self.Match(s,u_song)
                s_scores[s]+=math.pow(s_match,self.Q)/np
        for s in all_songs:
            if s in self.hsongs:
                s_scores[s]=self.calibrate(s_scores[s],s)
            else:
                s_scores[s]=0.0
        return s_scores        
    
class PredSU(Pred):

    '''Implement user-similarity based predictor'''
    
    def __init__(self, _u2s_tr, _A=0, _Q=1):
        Pred.__init__(self)
        self.u2s_tr = _u2s_tr
        self.Q = _Q
        self.A = _A
    
    def printati(self):
        print ("PredSU(A=%f,Q=%f)"%(self.A,self.Q),)

    def Score(self,user_songs,  all_songs):
        s_scores={}
        for u_tr in self.u2s_tr:
            if not u_tr in self.u2s_tr:
                continue
            w=float(len(self.u2s_tr[u_tr] & user_songs))
            if w > 0:
                l1=len(user_songs)
                l2=len(self.u2s_tr[u_tr])
                w/=(math.pow(l1,self.A)*(math.pow(l2,(1.0-self.A))))
                w=math.pow(w,self.Q)
            for s in self.u2s_tr[u_tr]:
                if s in s_scores:
                    s_scores[s]+=w
                else:
                    s_scores[s]=w
        return s_scores

###
### RECOMMENDERS
###

class Reco:

    '''Implements Recommender'''

    def __init__(self, _all_songs):
        self.predictors=[]
        self.all_songs=_all_songs
        self.tau=500 # he most likely 500 songs

    def Add(self,p):
        self.predictors.append(p) # p is the predSI object

class SReco(Reco):

    '''Implements Aggregated Stochastic Recommender'''

    def __init__(self,_all_songs):
        Reco.__init__(self,_all_songs)
        self.Gamma=[]
# picks an index which correspond to a predictor
    def GetStocIndex(self,n,distr):
        r=random.random()
        for i in range(n):
            if r<distr[i]:
                return i
            r-=distr[i]
        return 0
        #chooses a predictor based on the distr, and returns the list of songs recommended by the chosen predictor
    def GetStochasticRec(self,songs_sorted, distr):
        nPreds=len(self.predictors)
        r=[]
        ii = [0]*nPreds # [0,0,0,0]
        while len(r)<self.tau:
            pi = self.GetStocIndex(nPreds,distr)# picks an index which correspond to a predictor
            s = songs_sorted[pi][ii[pi]] # enumerates over the recommended songs by the chosen predictor
            if not s in r:
                r.append(s)
            ii[pi]+=1
        return r

    def Valid(self, T, users_te, u2s_v, u2s_h, n_batch=10):
        ave_AP=0.0
        for t in range(T):
            rusers = users_te[t*n_batch:(t+1)*n_batch]
            rec=[]
            start=time.clock()
            for i,ru in enumerate(rusers):
                if ru in u2s_v:
                    print ("%d] scoring user %s with %d songs"%(i,ru,len(u2s_v[ru])))
                else:
                    print ("%d] scoring user %s with 0 songs"%(i,ru))
                fl()
                songs_sorted=[]
                for p in self.predictors:
                    ssongs=[]
                    if ru in u2s_v:
                        ssongs=MSD_util.sort_dict_dec(p.Score(u2s_v[ru],self.all_songs))
                    else:
                        ssongs=list(self.all_songs)
                   
                    cleaned_songs = []
                    for x in ssongs:
                        if len(cleaned_songs)>=self.tau: 
                            break
                        if ru not in u2s_v or x not in u2s_v[ru]:
                             cleaned_songs.append(x)
                                            
                    songs_sorted+= [cleaned_songs]
                    
                rec += [self.GetStochasticRec(songs_sorted, self.Gamma)]

            cti=time.clock()-start
            print ("Processed in %f secs"%cti)
            fl()
            # valuta la rec cn la map
            map_cur = mAP(rusers,rec,u2s_h,self.tau)
            ave_AP+=map_cur
            print ("MAP(%d): %f (%f)"%(t,map_cur,ave_AP/(t+1)))
            print
            fl()
    
        print ("Done!")
# returns tau songs which are the most compatible to the user
    def RecommendToUser(self, user, u2s_v):
        songs_sorted=[]
        for p in self.predictors:
            ssongs=[]
            if user in u2s_v:
                ssongs=MSD_util.sort_dict_dec(p.Score(u2s_v[user],self.all_songs)) # Score returns dict (song from all_songs, score based on user history)
            else:
                ssongs=list(self.all_songs)

            cleaned_songs = []
            for x in ssongs:
                if len(cleaned_songs)>=self.tau: 
                    break # we only need tau songs for recommendation
                if x not in u2s_v[user]: # we don't want to recommend a song that the user has already listened to
                    cleaned_songs.append(x)

            songs_sorted += [cleaned_songs] #songs_sorted is an array (of #predictors) of an array (of recommended songs )

        return self.GetStochasticRec(songs_sorted, self.Gamma)         #chooses a predictor based on the distr, and returns the list of songs recommended by the chosen predictor

#l_users: set of users of kagger_users
#u2s_v: creates dict (user,songs which he has listened to) based on the evaluation set
    def RecommendToUsers(self, l_users, u2s_v):
        sti=time.clock()
        rec4users=[]
        for i,u in enumerate(l_users):
            if not (i+1)%10:
                if u in u2s_v: #9 out of 10 times comes here
                    print ("%d] %s w/ %d songs"%(i+1,l_users[i],len(u2s_v[u])),)
                else:
                    print ("%d] %s w/ 0 songs"%(i+1,l_users[i]),)
                fl()
            rec4users.append(self.RecommendToUser(u,u2s_v)) # returns tau songs which are the most compatible to the user
            cti=time.clock()-sti
            if not (i+1)%10:
                print (" tot secs: %f (%f)"%(cti,cti/(i+1)))
                fl()
                
        return rec4users
