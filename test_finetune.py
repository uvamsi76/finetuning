import torch
import numpy as np
from transformers import GPT2LMHeadModel
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getxys(buf,iter,B,T):
    
    if(B*T*(iter+1)+1>buf.shape[0]):
        pool=buf[B*T*iter:]+buf[:(B*T*(iter+1))-buf.shape[0]+1]    
    else:
        pool=buf[B*T*iter:B*T*(iter+1)+1]
    x=(pool[:-1]).view(B,T)
    y=(pool[1:]).view(B,T)
    return x,y

buf=np.load('numpy_arrays/data.npy')
buf = buf.astype(np.int32)
buf=torch.tensor(buf,dtype=torch.long)

train_buf=buf[:int(buf.shape[0]*0.8)]
val_buf=buf[int(buf.shape[0]*0.8):int(buf.shape[0]*0.9)]
test_buf=buf[int(buf.shape[0]*0.9):]

model_hf=GPT2LMHeadModel.from_pretrained('gpt2')

B=1
T=512
max_iter=20846 #buf.shape[0]//(B*T)

model_hf=model_hf.to(device)

optimizer=AdamW(params= model_hf.parameters(),lr=6e-5)

val_iter=0
for iter in range(max_iter):
    model_hf.train()
    optimizer.zero_grad()
    x,y=getxys(train_buf,iter,B,T)
    x=x.to(device)
    y=y.to(device)
    op=model_hf(input_ids=x,labels=y)
    # op.loss.retain_grad()
    op.loss.backward()
    optimizer.step()
    print(f'train_loss at iter {iter}:{op.loss.detach():.4f}')
    del x,y
    torch.cuda.empty_cache()
    if(iter%100==0):
        x_val,y_val=getxys(val_buf,val_iter,B,T)
        x_val,y_val=x_val.to(device),y_val.to(device)
        val_iter+=1
        model_hf.eval()
        op=model_hf(input_ids=x_val,labels=y_val)
        print(f'val_loss : {op.loss.detach():.4f}')

model_hf._save_to_state_dict('model/model_dict.pth')