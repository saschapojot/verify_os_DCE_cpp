import pandas as pd
from pathlib import Path
import sys

#python readCSV.py groupNum rowNum
if len(sys.argv)!=3:
    print("wrong number of arguments")

group=int(sys.argv[1])
rowNum=int(sys.argv[2])

inParamFileName="./inParamsNew"+str(group)+".csv"
# print("file name is "+inParamFileName)
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]

j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

g0=float(oneRow.loc["g0"])
omegam=float(oneRow.loc["omegam"])
omegap=float(oneRow.loc["omegap"])
omegac=float(oneRow.loc["omegac"])
er=float(oneRow.loc["er"])#magnification

thetaCoef=float(oneRow.loc["thetaCoef"])

print("j1H"+str(j1H)+"j2H"+str(j2H)+"g0"+str(g0)\
      +"omegam"+str(omegam)+"omegap"+str(omegap)+"omegac"+str(omegac)+"er"+str(er)+"thetaCoef"+str(thetaCoef))


