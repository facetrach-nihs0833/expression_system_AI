from yaspin import yaspin
import os
import re



files=os.listdir("train")
print(files)

images=[]  
labels=[]
path = 'output.txt'

with yaspin(text="LOAD image", color="cyan") as sp:
    for i in range(len(files)):


        file=os.listdir("train/"+files[i])
        for j in range(len(file)):
            os.rename(f"train/"+files[i]+"/"+file[j],"train/"+files[i]+"/"+re.sub(r"\s+", "", file[j]))         



    sp.write("complit")
    sp.ok("✔")
