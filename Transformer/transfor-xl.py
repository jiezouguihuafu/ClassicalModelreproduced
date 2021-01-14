from transformers import TransfoXLConfig,TransfoXLModel
import torch

d_model = 10
d_inner = 20
n_layer = 1

batch_size = 2
seq_len = 100

configuration = TransfoXLConfig(d_model=d_model,d_inner=d_inner,n_layer=n_layer)
model = TransfoXLModel(configuration)

output,atten=model(torch.ones(batch_size,seq_len,dtype=torch.int64))
print(output.shape) # [batch,seq_len,d_model]
print(atten[0].shape) # [mem_len,batch,d_model]


