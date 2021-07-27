from yaspin import yaspin
import os
import re



files=os.listdir("train")
print(files)

images=[]  
labels=[]
path = 'output.txt'


for i in range(len(files)):


    file=os.listdir("train/"+files[i])
    print(f"NAME:{files[i]},count:{len(file)}")


