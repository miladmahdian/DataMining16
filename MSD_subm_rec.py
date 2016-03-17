### MSD_subm_rec.py: Example of a recommendation script 
### Copyright (C) 2012  Fabio Aiolli                                                     

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details. 

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.             

# Scientific results produced using the software provided shall
# acknowledge the use of this software. Please cite as                                                            

#       F. Aiolli, A Preliminary Study on a Recommender System for the Million Songs Dataset Challenge
#       Preference Learning: Problems and Applications in AI (PL-12), ECAI-12 Workshop, Montpellier
#       http://www.ke.tu-darmstadt.de/events/PL-12/papers/08-aiolli.pdf                                           

# Moreover shall the author be informed about the
# publication.

import sys
import MSD_util,MSD_rec

#user_min,user_max,osfile=sys.argv[1:]
user_min=10 #int(user_min)
user_max=100 #int(user_max)
# path to the outpuut file kaggle_songs.txt
osfile = "output.txt"
print ("user_min: %d , user_max: %d"%(user_min,user_max))
sys.stdout.flush() #forces it to "flush" the buffer, meaning that it will write everything in the buffer to the terminal

# TRIPLETS
f_triplets_tr="train_triplets.txt" #48373586 triplets for training with exclusive users from kaggle_visible
f_triplets_tev="kaggle_visible_evaluation_triplets.txt" #1450933 triplets for recommendation evaluation, with exclusive new users users 

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

print ('Creating predictor..')

_A = 0.15
_Q = 3
### calibrated
### pr=MSD_rec.PredSIc(s2u_tr, _A, _Q, "songs_scores.txt")

### uncalibrated song-based predictor
pr=MSD_rec.PredSI(s2u_tr, _A, _Q)

print ('Creating recommender..')
cp = MSD_rec.SReco(songs_ordered) # the input songs to the recommender is from the train_triplets.
cp.Add(pr)
cp.Gamma=[1.0] # the prob. on how to choose different predictors, here we only have one predictor so it's just [1.0]

r=cp.RecommendToUsers(users_v[user_min:user_max],u2s_v)
MSD_util.save_recommendations(r,"kaggle_songs.txt",osfile)




