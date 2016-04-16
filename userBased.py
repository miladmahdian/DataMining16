import math;
all_triplets = [];
fileName = "kaggle_visible_evaluation_triplets.txt";
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
            
##folds
folds = 5
avgMAE = 0.0

#writing test and train files per fold
for i in range(0, folds):
    print ("Creating Testing and Training file for fold: "+  str(i+1));
    song_userMapping={}
    training_file = open('training.txt','w+');
    testing_file = open('testing.txt','w+');
    for k in range (0, len(users_to_triplets)):
        triplets_of_user = users_to_triplets[str(k)]
        length_of_user_triplets = len(triplets_of_user)
        foldSize = length_of_user_triplets / folds
        start = i * foldSize;
        end = start + foldSize -1
        for j in range(0 , length_of_user_triplets):
            # testing
            if (j >= start and j <= end):
                testing_file.write(triplets_of_user[j]+'\n');
            else:
                training_file.write(triplets_of_user[j]+'\n');
    print ("Testing and Training file Created for fold: "+  str(i+1));
    
    training_file.close();
    testing_file.close();
    print("Creating User Attributes File For Training Data");   
    UserAttributes = open('UserAttributes.txt','w');     
    
    with open("training.txt","r") as file:
        userAttr='';
        prevUser ='';
        for line in file:
            user,song,count=line.strip().split('\t');
            if(prevUser == user or prevUser == ''):        
                userAttr += song + ':' + count +',';
            else:
                UserAttributes.write(prevUser + '\t' +userAttr+'\n');
                userAttr = song + ':' + count +',';
                
            if song not in song_userMapping:
                userArray =[]
                userArray.append(user);
                song_userMapping[song] = userArray;
            else:
                userArray = song_userMapping[song];
                userArray.append(user);
                song_userMapping[song] = userArray;
            
            prevUser = user;
        UserAttributes.write(prevUser + '\t' +userAttr+'\n');
        UserAttributes.close();
            
    print("Created User Attributes File for Training Data");
    user_songMapping ={}
    
    print("Loading User Attributes File for Training Data in Dictionary");
    with open("UserAttributes.txt","r") as file1:
        for user_i in file1:
            user1, user1Attr = user_i.strip().split('\t');
            user_songMapping[user1] = user1Attr;
            
    print("Creating User Attributes File For Testing Data");   
    UserAttributes = open('UserAttributesTesting.txt','w');     
    
    with open("testing.txt","r") as file:
        userAttr='';
        prevUser ='';
        for line in file:
            user,song,count=line.strip().split('\t');
            if(prevUser == user or prevUser == ''):        
                userAttr += song + ':' + count +',';
            else:
                UserAttributes.write(prevUser + '\t' +userAttr+'\n');
                userAttr = song + ':' + count +',';
                
            prevUser = user;
        UserAttributes.write(prevUser + '\t' +userAttr+'\n');
        UserAttributes.close();
            
    print("Created User Attributes File for Testing Data");
    
    print("Loading Testing Data in a Dictionary");   
    testUserAttr ={};
    
    with open("UserAttributesTesting.txt","r") as file:
        for line in file:
                user,attributes=line.strip().split('\t')
                testUserAttr[int(user)] = attributes;
    
    print("Loaded User Attributes File for Training Data in Dictionary");
    
    sum =0;
    N =0;
    with open("testing.txt","r") as file:
        for line in file:
            N +=1;
            user1,song,count=line.strip().split('\t')
            Ruser1AvgCount=0.0;
            if song in song_userMapping:
                userArray = song_userMapping[song];
                num =0.0;
                den =0.0;
                for user2 in userArray:
                    user1Attr = user_songMapping[user1];
                    user2Attr = user_songMapping[user2];
                    P = commonSongs(user1Attr,user2Attr);
                    similarity = 0.0;
                    if(len(P)>0):
                        user1_songcounts = songCountMapping(user1Attr)
                        user2_songcounts = songCountMapping(user2Attr)
                        Ruser1AvgCount = averageRating(user1Attr);
                        Ruser2AvgCount = averageRating(user2Attr);
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
                            
                        num += similarity * (user2_songcounts[int(song)] - Ruser2AvgCount);
                        den += similarity;
                        
                prediction =0.0;
                
                if den !=0:
                    prediction = Ruser1AvgCount +  (num/den);      
                else:
                    prediction = Ruser1AvgCount; 
                    
                if prediction <0:
                    prediction = 0;
                    
                error = math.fabs(float(count) - prediction) 
                sum +=error
                
    foldMAE = sum / N
    print '\n\nMAE for fold\n\n '+ str(i) + ' -> ' + str(foldMAE)
    avgMAE += foldMAE
    
print 'Average MAE after 5 folds ' + str(avgMAE/5)
                    
