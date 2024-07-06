# import the necessary packages
import numpy as np
import pandas as pd
import subprocess
import os 
from Bio import SeqIO
from io import StringIO
from Bio import AlignIO
from Bio.Blast.Applications import NcbiblastnCommandline

# define the Needleman-Wunsch algorithm for determining the similarity between two sequences
def NW(seq1, seq2, matchscore = 1, mismatchscore = -1, gapscore = -1):
    l1 = len(seq1)
    l2 = len(seq2)
    A = np.zeros((l1 + 1, l2 + 1))
    # initialization of the matrix
    for i in range(1, l1 + 1):
        A[i][0] = i * gapscore
    for j in range(1, l2 + 1):
        A[0][j] = j * gapscore
    # dynamic programming to fill the matrix
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if seq1[i-1] == seq2[j-1]:
                A[i][j] = max(A[i-1][j-1]+matchscore, A[i-1][j]+gapscore, A[i][j-1]+gapscore)
            else:
                A[i][j] = max(A[i-1][j-1]+mismatchscore, A[i-1][j]+gapscore, A[i][j-1]+gapscore)
    # return the similarity score if it is positive
    if A[l1][l2] > 0:
        return A[l1][l2]
    return 0

# define the movement vector for the CGR algorithm
def map(a):
    dict={'A': np.array([0,0]), 'T': np.array([1,0]), 'C': np.array([0,1]), 'G': np.array([1,1]), 'a': np.array([0,0]), 't': np.array([1,0]), 'c': np.array([0,1]), 'g': np.array([1,1])}
    return dict[a]

# define the Chaos Game Representation (CGR) algorithm
def CGR(seq, N):
    # tem: initialization of the CGR matrix
    tem = np.zeros((N, N))
    n = len(seq)
    # X: initialization of the starting point
    X = np.array([0.5, 0.5])
    # move the point according to the sequence
    for i in range(n):
        if seq[i] == '-' or seq[i] == 'N' or seq[i] == 'n':
            continue
        tX = map(seq[i])
        X = 0.5 * X + 0.5 * tX
        tem[int(X[0] * N)][int(X[1] * N)] += 1
    return tem

# combine the CGR matrices of homologous sequences
def CGR_combine(homo, N):
    # combine: initialization of the combined CGR matrix
    combine = np.zeros((N, N))
    # get the combined CGR matrix based on the weight of similarity
    for i in range(len(homo)):
        combine += homo[i][1] * CGR(homo[i][0], N) / 50
    return combine

# generate the combined CGR matrix
def matrix_generation(filename, N):
    print("Generating matrix of " + filename)
    query = np.load("data\\" + filename + "_seq.npy")
    print("Sequence number", len(query))
    target = []
    for seq in SeqIO.parse("data\\blast_promoter50.fasta", "fasta"):
        target.append(str(seq.seq).lower())
    result = pd.read_table("data\\" + filename + ".table", header = None)
    qnum = result[0]
    tnum = result[1]
    Matchnum = 0
    homo = []
    for i in range(0, len(query)):
        tem = [[query[i], 50]]
        homo.append(tem)
    for i in range(0, len(qnum)):
        s = NW(query[int(qnum[i][1:])], target[int(tnum[i][1:])])
        if s <= 0:
            continue
        Matchnum += 1
        combine = [target[int(tnum[i][1:])], s]
        homo[int(qnum[i][1:])].append(combine)
    
    result = []
    for i in range(0, len(query)):
        PSS = CGR_combine(homo[i], N)
        result.append(PSS)
    np.save("data\\" + filename + "_matrix.npy", result) 
    print("Match number", Matchnum)
    print("Done")
