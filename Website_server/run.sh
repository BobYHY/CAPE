#!/bin/bash
original_seq=$1
mutation_start=$2
mutation_end=$3
mutated_base_num=$4
mutated_num=$5
selected_num=$6
echo ">m1" > original_seq.fasta
echo "$original_seq" >> original_seq.fasta
blastn -query original_seq.fasta -db blast_promoter50 -task blastn-short -evalue 1 -outfmt 6 -out similar.table
python Run.py "$original_seq" "$mutation_start" "$mutation_end" "$mutated_base_num" "$mutated_num" "$selected_num"
