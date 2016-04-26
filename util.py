
import sys



tr = "train_triplets_new.txt"
a = {}
with open(tr,"r") as read:
    for line in read:
        _,_,count=line.strip().split('\t')
        if count in a:
            a[count] += 1
        else:
            a[count] = 1
        
with open("dist.txt","w") as f:
    for key,value in a.iteritems():
        f.write(key+"\t"+str(value)+"\n")
        
f.close();
#print("max is " + str(max)) 
            
#sys.stdout.flush()
