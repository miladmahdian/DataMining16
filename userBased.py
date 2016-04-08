import math;
users = {};
songs ={};

training = open('training.txt','w');
testing = open('testing.txt','w');

print("Mapping each user and song to unique Index");
with open("kaggle_visible_evaluation_triplets.txt","r") as f:
        user_counter=0;
        song_counter=0;
        lines = '';
        for line in f:
            user,song,count=line.strip().split('\t')
            if song not in songs:
                songs[song] = song_counter;
                song_counter = song_counter+1;
                
            if user not in users:
                users[user] = user_counter;
                user_counter = user_counter+1;
                
                length = lines.count('\n');
                train_users = (length *4)/5;
                strArray = lines.split('\n');
                for i in range(0,train_users+1):
                    if len(strArray[i])>0:
                        training.write(strArray[i] + '\n');
                for i in range(train_users+1,length):
                    testing.write(strArray[i]+'\n');
                lines ='';
            
            lines += str(users[user]) +'\t'+ str(songs[song]) + '\t' + count+'\n';
            
        print("Created Tested and Training Files");
        training.close();
        testing.close();

def averageRating(userAttributes):
    songcountpairs = userAttributes.split(',');
    length = userAttributes.count(':');
    sum =0.0;
    for songcountpair in songcountpairs:
        if songcountpair != '':
            pair = songcountpair.split(':');
            sum += int(pair[1]);
    
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
    
def songCountMapping(userAttributes):
    songcountpairs = userAttributes.split(',');
    dict ={};
    for songcountpair in songcountpairs:
        if songcountpair != '':
            pair = songcountpair.split(':');
            dict[int(pair[0])] = int(pair[1]);
    
    return dict;

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
             
        prevUser = user;
    UserAttributes.write(prevUser + '\t' +userAttr+'\n');
    UserAttributes.close();
        
print("Created User Attributes File for Traing Data");

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
    
predictionFile = open('predictionFile.txt','w');

print("Defining Similarity Between Users");
with open("UserAttributes.txt","r") as file1:
    for user_i in file1:
         user1, user1Attr = user_i.strip().split('\t');
         if int(user1) not in testUserAttr:
             continue;
         testingData = songCountMapping(testUserAttr[int(user1)]);
         num ={};
         den ={};
         for key in testingData:
            num[key] = -averageRating(user1Attr);
            den[key] =0.0;
         with open("UserAttributes.txt","r") as file2:
             for user_j in file2:
                user2, user2Attr = user_j.strip().split('\t');
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
                        num1 += devuser1* devuser2;
                        user1den += math.pow(devuser1,2);
                        user2den += math.pow(devuser2,2);
                    
                    user1den = math.pow(user1den,0.5);
                    user2den = math.pow(user2den,0.5);
                    deno = user1den * user2den;
                    if(deno !=0 and num1!=0):
                        similarity = num1/deno;
                        #similarityFile.write(user1 +','+user2+'\t'+str(similarity)+'\n');

                    testingData = songCountMapping(testUserAttr[int(user1)]);  
                    for key in testingData:
                        if key in user2_songcounts:
                            num[key] += similarity * (user2_songcounts[key] - Ruser2AvgCount)
                            den[key] += similarity;
                    
             pred = 0;
             testingData = songCountMapping(testUserAttr[int(user1)]);
             for key in testingData: 
                if(den[key] !=0 and num[key]!=0):                       
                    pred = num[key]/den[key] + averageRating(user1Attr);
                print 'Prediction: '+user1 +' , '+str(key)+'\t'+str(pred)+'\n';
                predictionFile.write(user1 +' , '+str(key)+'\t'+str(pred)+'\n');
                
                        
predictionFile.close();
print('Done');        
                
                
                
                
                        
        




            


                
