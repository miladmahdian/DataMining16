import math

#run this script by python2.7
#the input is the dataset "train_triplets.txt"
#the output is the pruned dataset "train_triplets_concise.txt"


users_to_triplets = {}
with open("train_triplets.txt","r") as f:
        lines = '';
        for line in f:
            user,song,count=line.strip().split('\t')
            if user in users_to_triplets:
                triplets_of_user = users_to_triplets[user]
                triplets_of_user.append( song + '\t' + count)
                users_to_triplets[user] = triplets_of_user
            else:
                triplets_of_user = []
                triplets_of_user.append(song + '\t' + count)
                users_to_triplets[user] = triplets_of_user
     


n_triplets = 0
user_counter = 0
song_counter = 0
users = {};
songs ={};
output = open('train_triplets_concise.txt','w');
for user,triplets_of_user in users_to_triplets.iteritems():
    length_of_user_triplets = len(triplets_of_user)
    if length_of_user_triplets < 200:
        continue
    else:
        
        for line in triplets_of_user:
            n_triplets+= 1 
            song,count=line.strip().split('\t')
            if song not in songs:
                songs[song] = song_counter;
                song_counter = song_counter+1;
                
            if user not in users:
                users[user] = user_counter;
                user_counter = user_counter+1;

            output.write( str(users[user]) +'\t'+ str(songs[song]) + '\t' + count+'\n')
                
print("Created Training File");
output.close();

