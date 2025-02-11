import pandas as pd
import glob
import numpy as np
import tiktoken

df=pd.read_csv('master.csv')
df2=pd.concat([df['Question']+df['Answer']],ignore_index=True)
df2=" {Question} : "+df['Question']+" {Answer} : "+df['Answer']
df2.iloc[0]
df2.shape
txt=''
for i in range(df2.shape[0]):
    txt=txt+df2.iloc[i]
enc=tiktoken.get_encoding('gpt2')
tokens=[]
tokens.extend(enc.encode(txt))
tokens_np=np.array(tokens)
tokens_np_uint16=tokens_np.astype(np.uint16)
np.save('numpy_arrays/data',tokens_np_uint16)