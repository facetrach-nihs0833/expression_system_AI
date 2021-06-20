from yaspin import yaspin
import os




files=os.listdir("train")
print(files)

images=[]  
labels=[]
path = 'output.txt'

with yaspin(text="LOAD image", color="cyan") as sp:
    f = open(path, 'w')
    for i in range(len(files)):


        file=os.listdir("train/"+files[i])
        for j in range(len(file)):
            images.append(files[i]+"/"+file[j]+" "+str(i)+"\n")
            labels.append(i)

            if j >=15000:
                break

    f.writelines(images)
    f.close()
    sp.write("complit")
    sp.ok("✔")
