import pyprind
import pandas as pd
import os
import numpy as np
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
            df=df.append([[txt,labels[l]]],ignore_index=True)
            pbar.update()
df.columns=['review','sentiment']
print(df)
#Now adding the code for exporting to csv. We don't want to spend 10 mins loading it every time.
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv',index=False)
