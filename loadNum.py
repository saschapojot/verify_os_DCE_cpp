import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import pandas as pd
import sys
import re
from pathlib import Path
import pyarma as arma
#this script loads computed particle numbers
if len(sys.argv)!=3:
    print("wrong number of arguments")

groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])




inParamFileName="./inParamsNew"+str(groupNum)+".csv"
#read parameters from csv
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]

j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

g0=float(oneRow.loc["g0"])
omegam=float(oneRow.loc["omegam"])
omegap=float(oneRow.loc["omegap"])
omegac=float(oneRow.loc["omegac"])
er=float(oneRow.loc["er"])#magnification
r=np.log(er)
thetaCoef=float(oneRow.loc["thetaCoef"])
theta=thetaCoef*np.pi
Deltam=omegam-omegap
e2r=er**2
lmd=(e2r-1/e2r)/(e2r+1/e2r)*Deltam

inDir="./groupNew"+str(groupNum)+"/row"+str(rowNum)+"/"
N1=0
N2=0
L1=0
L2=0
flsNums=[]
fileNames=[]
for file in glob.glob(inDir+"/*.json"):
    fileNames.append(file)
    matchFlush=re.search(r"flush(\d+)N1",file)
    if matchFlush:
        flsNums.append(int(matchFlush.group(1)))
    matchN1=re.search(r"N1(\d+)N2",file)
    if matchN1:
        N1=int(matchN1.group(1))
    matchN2=re.search(r"N2(\d+)L1",file)
    if matchN2:
        N2=int(matchN2.group(1))
    matchL1=re.search(r"L1(\d+(\.\d+)?)L2",file)
    if matchL1:
        L1=float(matchL1.group(1))

    matchL2=re.search(r"L2(\d+(\.\d+)?)",file)
    if matchL2:
        L2=float(matchL2.group(1))

flushInds=np.argsort(flsNums)
sortedFileNames=[fileNames[i] for i in flushInds]



def readFiles(firstFileInd,lastFileInd):
    """

    :param firstFileInd: first file's flush
    :param lastFileInd: last file's flush
    :return: the particle numbers
    """
    filesToRead=[sortedFileNames[i] for i in range(firstFileInd,lastFileInd+1)]


    photonNumsAll=[]
    phononNumsAll=[]

    filesLength=len(filesToRead)

    for j in range(0,filesLength-1):
        fileTmp=filesToRead[j]
        with open(fileTmp) as json_fptr:
            dataTmp=json.load(json_fptr)
            photonNum=dataTmp["photonNum"]
            phononNum=dataTmp["phononNum"]
            numLenTmp=len(photonNum)
            for n in range(0,numLenTmp-1):
                photonNumsAll.append(photonNum[n])
                phononNumsAll.append(phononNum[n])

    fileLast=filesToRead[filesLength-1]
    with open(fileLast) as json_fptr:
        dataTmp=json.load(json_fptr)
        photonNumsAll+=dataTmp["photonNum"]
        phononNumsAll+=dataTmp["phononNum"]

    return photonNumsAll,phononNumsAll

flsStart=0
flsEnd=2999

photonNumsAll,phononNumsAll=readFiles(flsStart,flsEnd)
#outdir
path0=inDir+ str(groupNum)+"/num/both/"
path1=inDir+str(groupNum)+"/num/photon/"
path2=inDir+str(groupNum)+"/wv/"
Path(path0).mkdir(parents=True, exist_ok=True)
Path(path1).mkdir(parents=True, exist_ok=True)
Path(path2).mkdir(parents=True, exist_ok=True)


tFlushStart=0
tFlushStop=0.001
tTotPerFlush=tFlushStop-tFlushStart

flushTimeStart=flsStart*tTotPerFlush
flushTimeEnd=flsEnd*tTotPerFlush

dtEst=0.0001
stepsPerFlush=int(np.ceil(tTotPerFlush/dtEst))
dt=tTotPerFlush/stepsPerFlush

timeGrids=[flushTimeStart+dt*j for j in range(0,len(photonNumsAll))]

plt.plot(timeGrids,photonNumsAll,color="blue",label="photon")
plt.plot(timeGrids,phononNumsAll,color="red",label="phonon")

tInterval=flushTimeEnd-flushTimeStart

xTicks=[flushTimeStart+j/4*tInterval for j in range(0,5)]
plt.xticks(xTicks)
plt.title("$g_{0}=$"+str(g0)+", $\omega_{c}=$"+str(omegac)+", $e^{r}=$"+str(er))
plt.xlabel("time")
plt.ylabel("number")
plt.legend(loc="upper left")
plt.savefig(path0+"flush"+str(flsStart)+"toflush"+str(flsEnd)+"group"+str(groupNum)+"row"+str(rowNum)+"bothNum.png")
plt.close()

plt.plot(timeGrids,photonNumsAll,color="blue",label="photon")
plt.xticks(xTicks)
plt.title("$g_{0}=$"+str(g0)+", $\omega_{c}=$"+str(omegac)+", $e^{r}=$"+str(er))
plt.xlabel("time")
plt.ylabel("number")
plt.legend(loc="upper left")
plt.savefig(path1+"flush"+str(flsStart)+"toflush"+str(flsEnd)+"group"+str(groupNum)+"row"+str(rowNum)+"photon.png")
plt.close()


wvFunctionInit=""
wvFunctionFinal=""
for file in glob.glob(inDir+"/*.txt"):
    matchInit=re.search(r"init",file)
    if matchInit:
        wvFunctionInit=file
    matchFinal=re.search(r"final",file)
    if matchFinal:
        wvFunctionFinal=file


psiFinal=arma.cx_mat()
psiFinal.load(wvFunctionFinal,arma.raw_ascii)
psiFinal=np.abs(np.array(psiFinal))

plt.figure()
plt.imshow(psiFinal)
plt.title("finalfls"+str(flsEnd))
plt.colorbar()
plt.savefig(path2+"finalfls"+str(flsEnd)+".png")
# plt.close()

psiInit=arma.cx_mat()
psiInit.load(wvFunctionInit,arma.raw_ascii)
psiInit=np.abs(np.array(psiInit))
plt.figure()
plt.imshow(psiInit)
plt.title("init")
plt.colorbar()
plt.savefig(path2+"init.png")