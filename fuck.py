from yaspin import yaspin
import os
import re



files=os.listdir("fer2013")
print(files)

images=[]  
labels=[]
path = 'output.txt'

with yaspin(text="LOAD image", color="cyan") as sp:
    f = open(path, 'w')
    for i in range(len(files)):


        file=os.listdir("fer2013/"+files[i])
        for j in range(len(file)):
            images.append(files[i]+"/"+re.sub(r"\s+", "", file[j])+" "+str(i)+"\n")
            labels.append(i)

            # if j >=2000:
            #     break

    f.writelines(images)
    f.close()
    sp.write("complit")
    sp.ok("✔")
