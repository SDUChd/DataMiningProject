import json
import codecs as cs
import os               # 导入os模块


taskList=[]
with cs.open("cna_test_unass_competition.json","r") as f:
    taskList=json.loads(f.read())
    for i in range(len(taskList)):
        taskList[i]=taskList[i].split("-")
    print(taskList)

paperList=[]
with cs.open("cna_test_pub.json","r") as f:
    paperList=json.loads(f.read())
    # print(paperList)

wholeAuthor=[]

with cs.open("whole_author_profile.json","r") as f:
    wholeAuthor=json.loads(f.read())


wholePapers=[]
with cs.open("whole_author_profile_pub.json","r") as f:
    wholePapers=json.loads(f.read())

taskListDecode=[]

for t in taskList:
    authname = paperList[t[0]]["authors"][int(t[1])]["name"].replace("\u00a0"," ")
    authname=authname.split(" ")
    firstname=authname[0].lower()
    if len(authname)>1:
        if "-" in authname[1]:
            lastname=authname[1].split("-")
            lastname=lastname[0].lower()+lastname[1].lower()
        else:
            lastname=authname[1].lower()
        authname=lastname+"_"+firstname
    else:
        authname=authname[0].lower()
    papertitle=paperList[t[0]]["title"]
    try:
        keywords=paperList[t[0]]["keywords"]
    except:
        keywords=[]
    taskListDecode.append([authname,papertitle,keywords,t[0]])

count=0
print("start!")
resstr=""

initRes=[]
with cs.open("submit.json","r") as f:
    initRes=json.loads(f.read())

finalRes=[]
with cs.open("res3.txt","w") as f:
    for t in taskListDecode:
        potentialAuth=[]
        count += 1
        for k,v in wholeAuthor.items():
            if v['name']==t[0]:
                onepotential=[]
                for op in v["papers"]:
                    p=wholePapers[op]
                    title=p["title"]
                    try:
                        keywords=p["keywords"]
                    except:
                        keywords=[]
                    onepotential.append([title,keywords,k])
                potentialAuth.append(onepotential)

        if len(potentialAuth)>=10 and len(potentialAuth)<15:
            f.write(str(count / len(taskListDecode))+"\n")
            f.write("待分配论文信息"+"\n")
            f.write(t[0]+"\n")
            f.write(t[1]+"\n")
            f.write(str(t[2])+"\n")
            f.write(t[3]+"\n")
            f.write("候选人论文信息："+"\n")
            for onecandidate in potentialAuth:
                f.write("--------"+onecandidate[0][2]+"----------"+"\n")
                for w in onecandidate:
                    f.write(w[0]+"\n")
                    f.write(str(w[1])+"\n")
                    break
                f.write("------------------"+"\n")

            pindex=t[3]
            hasAssined = False
            for k, v in initRes.items():
                for w in v:
                    if pindex == w:
                        hasAssined = True
                        break
            if not hasAssined:
                finalRes.append(pindex)
            # print(count)



with cs.open("finalRes_long.txt","w") as f:
    for t in finalRes:
        f.write(t+"\n")




# print (taskListDecode)