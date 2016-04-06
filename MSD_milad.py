
import sys
import MSD_util,MSD_rec

# to make new test and train sets out of train_triplet
#### ATTENTION
# run this one time to create the files and then comment it
#MSD_util.make_test_file2()


#user_min,user_max,osfile=sys.argv[1:]
#user_min=10 #int(user_min)
#user_max=100 #int(user_max)
# path to the outpuut file kaggle_songs.txt
osfile = "output.txt"
#print ("user_min: %d , user_max: %d"%(user_min,user_max))
sys.stdout.flush() #forces it to "flush" the buffer, meaning that it will write everything in the buffer to the terminal

# TRIPLETS
f_triplets_tr="train.txt" #48373586 triplets for training with exclusive users from kaggle_visible
f_triplets_tev="testV.txt" #1450933 triplets for recommendation evaluation, with exclusive new users users 
f_triplets_teh = "testH.txt"
print ('loading users in %s'%"kaggle_users.txt")
sys.stdout.flush()
users_v=list(MSD_util.load_users("kaggle_users.txt"))

print ('default ordering by popularity')
sys.stdout.flush()
songs_ordered=MSD_util.sort_dict_dec(MSD_util.song_to_count(f_triplets_tr)) # song_to_count creates a dictionary (song,count) and then it sorts the dict in decresing order

print  ("loading unique users indexes")
uu = MSD_util.unique_users(f_triplets_tr) #unique_users returns a set of unique users in the train_triplets


u2i = {} # creates a dictionary (userId,index)
for i,u in enumerate(uu):
    u2i[u]=i

print ('song to users on %s'%f_triplets_tr)
s2u_tr=MSD_util.song_to_users(f_triplets_tr) #creates dict with (song, set of users who have listened to this song)

print ("converting users to indexes") #converts the userIDs in s2u_tr to their index uu
for s in s2u_tr:
    s_set = set()
    for u in s2u_tr[s]:
        s_set.add(u2i[u])
    s2u_tr[s]=s_set

del u2i

print ('user to songs on %s'%f_triplets_tev)
u2s_v=MSD_util.user_to_songs(f_triplets_tev) #creates dict (user,songs which he has listened to) based on the evaluation set
u2s_h = MSD_util.user_to_songs(f_triplets_teh) #hidden half of the evaluation
print ('Creating predictor..')
_A = 0.15
_Q = 3
### calibrated
### pr=MSD_rec.PredSIc(s2u_tr, _A, _Q, "songs_scores.txt")

### uncalibrated song-based predictor
pr=MSD_rec.PredSI(s2u_tr, _A, _Q)
users_te = MSD_util.unique_users(f_triplets_teh) #users we want to recommend songs to
print("Number of users_te: "+str(len(users_te)))
print ("Size of intersection between test users and training users: "+str(len(uu & users_te)))
print ('Creating recommender..')
#ave_AP=0.0
#n_batch=10
cp = MSD_rec.SReco(songs_ordered) # the input songs to the recommender is from the train_triplets.
cp.Add(pr)
cp.Gamma=[1.0] # the prob. on how to choose different predictors, here we only have one predictor so it's just [1.0]

cp.Valid(10000, list(users_te),u2s_v,u2s_h) # this does the validation  
#r=cp.RecommendToUsers(users_v[user_min:user_max],u2s_v)
#MSD_util.save_recommendations(r,"kaggle_songs.txt",osfile) #saves the output in terms of 500 song indices for each user seperated by space and then each user by \n




