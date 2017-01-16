import pyprind
import pandas as pd
import os

basepath='aclImdb'

labels={'pos':1,'neg':0}
pbar=pyprind.ProgBar(50000)
df=pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path=os.path.join(basepath,s,l)
        # listing the files in directory
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                # read only mode
                txt=infile.read()
            df=df.append([txt,labels[l]],ignore_index=True)
            pbar.update()
# df.columns=['review','sentiment']
