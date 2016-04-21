# -*- coding: cp1252 -*-
import math;
import time
all_triplets = [];
fileName = "train_triplets.txt";
songs = {}
users = {}
users_to_triplets = {}

start_time = time.localtime()
print start_time
print("Mapping each user and song to unique Index and creating a user to triplets dictionary");

# creating a dictionary with key = user and value = array of this user's entries
# this dictionary contains the whole train_triplets data
with open(fileName,"r") as f:
        user_counter=0;
        song_counter=0;
        lines = '';
        for line in f:
            user,song,count=line.strip().split('\t')
            count = 1+math.log(int(count))
            if song not in songs:
                songs[song] = song_counter;
                song_counter = song_counter+1;
                
            if user not in users:
                users[user] = user_counter;
                user_counter = user_counter+1;
            
            if str(users[user]) in users_to_triplets:
                triplets_of_user = users_to_triplets[str(users[user])]
                triplets_of_user.append(str(users[user]) + '\t' + str(songs[song]) + '\t' + str(count))
                users_to_triplets[str(users[user])] = triplets_of_user
            else:
                triplets_of_user = []
                triplets_of_user.append(str(users[user]) + '\t' + str(songs[song]) + '\t' + str(count))
                users_to_triplets[str(users[user])] = triplets_of_user
            


# pruning criteria - include only those users who have listened to at least 200 songs
for k in range (0, len(users_to_triplets)):
    triplets_of_user = users_to_triplets[str(k)]
    length_of_user_triplets = len(triplets_of_user)
    if length_of_user_triplets < 200:
        continue
    else:
        # do your work
