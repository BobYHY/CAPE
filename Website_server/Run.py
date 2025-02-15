from model.promoter_model import PrompterModel
import torch
import torch.nn as nn
import numpy as np
import sys
import pandas as pd
from Bio import SeqIO
import os

def NW(seq1,seq2,matchscore=1,mismatchscore=-1,gapscore=-1):
    l1=len(seq1)
    l2=len(seq2)
    A=np.zeros((l1+1,l2+1))
    for i in range(1,l1+1):
        A[i][0]=i*gapscore
    for j in range(1,l2+1):
        A[0][j]=j*gapscore
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            if seq1[i-1]==seq2[j-1]:
                A[i][j]=max(A[i-1][j-1]+matchscore,A[i-1][j]+gapscore,A[i][j-1]+gapscore)
            else:
                A[i][j]=max(A[i-1][j-1]+mismatchscore,A[i-1][j]+gapscore,A[i][j-1]+gapscore)
    if A[l1][l2]>0:
        return A[l1][l2]
    return 0

def map(a):
    dict={'A':np.array([0,0]),'T':np.array([1,0]),'C':np.array([0,1]),'G':np.array([1,1]),'a':np.array([0,0]),'t':np.array([1,0]),'c':np.array([0,1]),'g':np.array([1,1])}
    return dict[a]

def CGR(seq,N=20):
    tem=np.zeros((N,N))
    n=len(seq)
    X=np.array([0.5,0.5])
    for i in range(n):
        if seq[i]=='-' or seq[i]=='N' or seq[i]=='n':
            continue
        tX=map(seq[i])
        X=0.5*X+0.5*tX
        tem[int(X[0]*N)][int(X[1]*N)]+=1
    return tem


def CGR_combine(homo,N=20):
    combine=np.zeros((N,N))
    for i in range(len(homo)):
        combine+=homo[i][1]*CGR(homo[i][0],N)/50
    return combine

def generator(original_seq, mutation_start, mutation_end, mutated_base_num, mutated_num): #start=0,end=50
    np.random.seed(0)
    res=[]
    res.append(original_seq.lower())
    i = 0
    while i < mutated_num: 
        tem=list(original_seq.lower())
        for j in range(mutated_base_num):
            r1=np.random.randint(mutation_start,mutation_end,1)[0]
            r2=np.random.randint(1,5,1)[0]
            if (r2==1):
                tem[r1]='a'
            if (r2==2):
                tem[r1]='g'
            if (r2==3):
                tem[r1]='c'
            if (r2==4):
                tem[r1]='t'
        res.append("".join(tem))
        i = i + 1
    return res


def main(original_seq, mutation_start, mutation_end, mutated_base_num, mutated_num, selected_num):
    model = PrompterModel(input_dim=100,
                            embedding_dim=32,
                            depth_transformer=2,
                            heads_transformer=8,
                            dim_head_transformer=64,
                            attn_dropout_transformer=0.1,
                            ff_dropout_transformer=0.1,
                            dropout_CNN=0.2,
                            mat_size=20
                            )
    Weight= torch.load("best_previous.pth",map_location="cpu")
    model.load_state_dict(Weight)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    Similar_seq=[]
    Similar_CGR=[]
    if os.path.getsize("similar.table") != 0:
        similartable=pd.read_table("similar.table",header=None)
        tnum=similartable[1]
        target=[]
        for seq in SeqIO.parse("promoter50.fasta", "fasta"):
            target.append(str(seq.seq).lower())
        for i in range(len(tnum)):
            if NW(original_seq, target[int(tnum[i][1:])])>-mutated_base_num: #!!!!!!!! should be checked
                Similar_seq.append(target[int(tnum[i][1:])])
                Similar_CGR.append(CGR(target[int(tnum[i][1:])]))
    wv = np.load("word2vec.npy", allow_pickle = True).item()
    sequencelist=generator(original_seq,mutation_start, mutation_end, mutated_base_num, mutated_num)
    N = mutated_num + 1 #including original
    Y=np.zeros(N)
    Batch=1
    for i in range(int(N/Batch)):
        Batch_CGR=[]
        Batch_X=[]
        for l in range(Batch):
            id=i*Batch+l
            s=sequencelist[id]
            s_CGR=CGR(s)
            for j in range(len(Similar_seq)):
                score=NW(s,Similar_seq[j])
                if score>0:
                    s_CGR+=Similar_CGR[j]*score/50
            Batch_CGR.append(s_CGR)
            sX = []
            for j in range(0, 48):
                sX.append(wv[s[j:j + 3]])
            Batch_X.append(sX)
        Batch_X = torch.from_numpy(np.array(Batch_X).astype(np.float32)).float()
        Batch_CGR=torch.from_numpy(np.array(Batch_CGR).astype(np.float32)).unsqueeze(1)
        y=model(Batch_X,Batch_CGR)
        yy=y.detach().numpy().reshape(-1)
        Y[i*Batch:(i+1)*Batch]=yy

    def get_top_n_indexes(arr, n):
        idx = np.argsort(arr)[-n:]
        return idx
    MAXLIST=get_top_n_indexes(Y,selected_num) 
    ans=[]
    for i in range(len(MAXLIST)):
        index=MAXLIST[len(MAXLIST)-i-1]
        ans.append((sequencelist[index],Y[index]))
    print("Original sequences",sequencelist[0],Y[0])
    print("Best mutated sequences",ans)
    return 

if __name__ == "__main__":
    original_seq = sys.argv[1]
    mutation_start = int(sys.argv[2]) 
    mutation_end = int(sys.argv[3]) 
    mutated_base_num = int(sys.argv[4]) 
    mutated_num = int(sys.argv[5]) 
    selected_num = int(sys.argv[6]) 
    main(original_seq, mutation_start, mutation_end, mutated_base_num, mutated_num, selected_num)
    # main("ctgaaattatttcgttgtacccacaggttcagtggtttcattattacctg", 0, 50, 5, 1000, 3)
