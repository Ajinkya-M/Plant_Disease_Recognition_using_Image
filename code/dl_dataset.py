import wget
import uuid
import shutil
import os

path = "csv files"

saveFolderPath = "data"

onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#print(onlyfiles)

if not os.path.exists(saveFolderPath):
    os.mkdir(saveFolderPath)

flag = 0
logfp = open("log.txt", "w")
for fname in onlyfiles :
    print("File : ", fname)
    logfp.write("File : %s\n" % (fname))
    
    with open("{0}/{1}".format(path, fname)) as fp:
        folderName = fname
	  
        if folderName.endswith("_images.csv"):
            folderName = folderName[:-11]
        
        if not os.path.exists(saveFolderPath + "/" + folderName):
            os.mkdir(saveFolderPath + "/" + folderName)
            j = 0
            total = 0
            for line in fp:
                total = total + 1
                try:                    
                    tok = line.split(",")
                    if line != "" and tok[0] == "Crop common name":
                        flag = 1
                        total = 0
                        continue
                    if flag == 1 and len(tok) > 4 and 'https' in tok[4]:                        
                        savedFile = wget.download(tok[4])
                        #print(j)
                        newFilePath = "{0}/{1}/{2}_{3}".format(saveFolderPath, folderName, uuid.uuid4(), savedFile)
                        shutil.move(savedFile, newFilePath)
                        j = j + 1
                except:
                    cnt = 0
                    while True:
                        try:
                            savedFile = wget.download(line.split(",")[4])
                            #print(j)
                            newFilePath = "{0}/{1}/{2}_{3}".format(saveFolderPath, folderName, uuid.uuid4(), savedFile)
                            shutil.move(savedFile, newFilePath)
                            j = j + 1
                            break
                        except:
                            cnt = cnt + 1
                            if cnt > 7:
                                #print(line.split(",")[4])
                                logfp.write(line.split(",")[4])
                                logfp.write("\n")
                                break
            print("Downloaded = %d / %d = %f" % (j, total, j / total))
                    
                    
    #break
logfp.close()
                    
			
