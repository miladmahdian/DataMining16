import math
import random
import time
start_time = time.localtime()
print start_time
fileName = "train_triplets.txt";
songs = {}
users = {}
users_to_triplets = {}

def songCountMapping(userAttributes):
    songcountpairs = userAttributes.split(',');
    dict ={};
    for songcountpair in songcountpairs:
        if songcountpair != '':
            pair = songcountpair.split(':');
            dict[int(pair[0])] = float(pair[1]);
    
    return dict;
    
def averageRating(userAttributes):
    songcountpairs = userAttributes.split(',');
    length = userAttributes.count(':');
    sum =0.0;
    for songcountpair in songcountpairs:
        if songcountpair != '':
            pair = songcountpair.split(':');
            sum += float(pair[1]);
    
    return sum/length;
    
def commonSongs(user1Attributes,user2Attributes):
    list1=[];
    list2=[];
    songcountpairs1 = user1Attributes.split(',');
    songcountpairs2 = user2Attributes.split(',');
    
    for songcountpair in songcountpairs1:
        if songcountpair != '':
            pair = songcountpair.split(':');
            list1.append(int(pair[0]));
    
    for songcountpair in songcountpairs2:
        if songcountpair != '':
            pair = songcountpair.split(':');
            list2.append(int(pair[0]));
            
    return list(set(list1).intersection(list2));


print("Mapping each user and song to unique Index and creating a user to triplets dictionary");
with open(fileName,"r") as f:
        user_counter=0;
        song_counter=0;
        lines = '';
        for line in f:
            user,song,count=line.strip().split('\t')
            count = 1 + math.log(int(count))
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

print ("Creating a pruned file")
pruned_file = open('PrunedFileUserBased.txt','w');                    
for k in range (0, len(users_to_triplets)):
    triplets_of_user = users_to_triplets[str(k)]
    length_of_user_triplets = len(triplets_of_user)
    if length_of_user_triplets < 200:
        continue
    for entry in triplets_of_user:
        user, song, count = entry.split('\t')
        pruned_file.write(str(user) + '\t' + str(song) + '\t' + str(count) + '\n')
print ("Created pruned file")
pruned_file.close()    

shuffledList = []
print ("Reading the lines of pruned file and shuffling")
lines = open('PrunedFileUserBased.txt').readlines()
random.shuffle(lines)

for line in lines:
    shuffledList.append(line)
    
userbased_output_file = open('OutputFileUserBased.txt','w');                        
print ("Now to the main part")
folds = 5
foldSize = len(shuffledList) / folds
avgMAE = 0.0
training_lines = 0
test_lines = 0




#writing test and train files per fold
for i in range(0, folds):
    print ("Creating Testing and Training file for fold: "+  str(i+1));
    song_userMapping={}
    training_file = open('training.txt','w+');
    testing_file = open('testing.txt','w+');
    start = i * foldSize;
    end = start + foldSize -1
    for j in range(0 , len(shuffledList)):
        # testing
        if (j >= start and j <= end):
            test_lines += 1
            testing_file.write(shuffledList[j]);
        else:
            training_lines += 1
            training_file.write(shuffledList[j]);
    print ("Testing and Training file Created for fold: "+  str(i+1));
    
    training_file.close();
    testing_file.close();
    print 'testing lines = ' + str(test_lines)
    print 'training lines = ' + str(training_lines)
    
    print("Creating User Attributes Dictionary For Training Data");   
    with open("training.txt","r") as file:
        train_user_songMapping ={}
        for line in file:
            user,song,count = line.strip().split('\t');
            if user in train_user_songMapping:
                attribute = train_user_songMapping[str(user)]
                attribute += song + ':' + count +',';
                train_user_songMapping[str(user)] = attribute
            else:
                train_user_songMapping[str(user)] = song + ':' + count +',';
                
            if song not in song_userMapping:
                userArray =[]
                userArray.append(user);
                song_userMapping[song] = userArray;
            else:
                userArray = song_userMapping[song];
                userArray.append(user);
                song_userMapping[song] = userArray;
    print("Created User Attributes Dictionary For Training Data");  
            
    print("Creating User Attributes Dictionary For Testing Data");     
    with open("testing.txt","r") as file:
        test_user_songMapping ={}
        for line in file:
            user,song,count = line.strip().split('\t');
            if user in test_user_songMapping:
                attribute = test_user_songMapping[str(user)]
                attribute += song + ':' + count +',';
                test_user_songMapping[str(user)] = attribute
            else:
                test_user_songMapping[str(user)] = song + ':' + count +',';
    print("Created User Attributes Dictionary For Testing Data");  
    
    sum = 0;
    N = 0;
    counter = 1
    with open("testing.txt","r") as file:
        for line in file:
            N += 1;
            simDic = {}
            counter += 1
            user1,song,count=line.strip().split('\t')
            Ruser1AvgCount=0.0;
            if song in song_userMapping:
                userArray = song_userMapping[song];
                num =0.0;
                den =0.0;
                for user2 in userArray:
                    user1Attr = train_user_songMapping[user1];
                    user2Attr = train_user_songMapping[user2];
                    Ruser2AvgCount = averageRating(user2Attr);
                    user2_songcounts = songCountMapping(user2Attr)
                    if (str(user1) + ',' + str(user2)) not in simDic:
                        P = commonSongs(user1Attr,user2Attr);
                        similarity = 0.0;
                        if(len(P)>0):
                            user1_songcounts = songCountMapping(user1Attr)
                            Ruser1AvgCount = averageRating(user1Attr);
                            num1 =0.0;
                            user1den = 0.0;
                            user2den = 0.0;
                            for p in P:
                                Ruser1songcount = user1_songcounts[p]
                                Ruser2songcount = user2_songcounts[p]
                                devuser1 =  Ruser1songcount - Ruser1AvgCount
                                devuser2 =  Ruser2songcount - Ruser2AvgCount
                                if devuser1 < 1:
                                    devuser1 = 1
                                if devuser2 < 1:
                                    devuser2 = 1
                                num1 += devuser1* devuser2;
                                user1den += math.pow(devuser1,2);
                                user2den += math.pow(devuser2,2);
                            
                            user1den = math.pow(user1den,0.5);
                            user2den = math.pow(user2den,0.5);
                            deno = user1den * user2den;
                            prevSim = similarity;
                            if(deno !=0 and num1!=0):
                                similarity = num1/deno;
                                
                            simDic[str(user1) + ',' + str(user2)] = similarity
                            
                            num += similarity * (user2_songcounts[int(song)] - Ruser2AvgCount);
                            den += similarity;
                    else:  
                        similarity = simDic[str(user1) + ',' + str(user2)]
                            
                        num += similarity * (user2_songcounts[int(song)] - Ruser2AvgCount);
                        den += similarity;
                
                prediction =0.0;
                
                if den !=0:
                    prediction = Ruser1AvgCount +  (num/den);      
                else:
                    prediction = Ruser1AvgCount; 
                    
                if prediction < 0:
                    prediction = 0;
                    
                error = math.fabs(float(count) - prediction) 
                sum +=error
                if counter%10000 == 0:
                    userbased_output_file.write('MAE after ' + str(counter) + ' test triplets ' + str(sum/counter) + '\n')
                    print 'MAE after ' + str(counter) + ' test triplets ' + str(sum/counter)
                    start_time = time.localtime()
                    print start_time
                
    foldMAE = sum / N
    userbased_output_file.write('\n\nMAE for fold\n\n '+ str(i) + ' -> ' + str(foldMAE))
    print '\n\nMAE for fold\n\n '+ str(i) + ' -> ' + str(foldMAE)
    avgMAE += foldMAE
    
userbased_output_file.write('\n' + 'Average MAE after 5 folds ' + str(avgMAE/5) + '\n')
print 'Average MAE after 5 folds ' + str(avgMAE/5)
    





