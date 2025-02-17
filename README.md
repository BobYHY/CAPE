# CAPE: a deep learning framework with Chaos-Attention net for Promoter Evolution

## Access to the website
For guidance on directed evolution for your sequence, please use our website [http://www.cape-promoter.com](http://www.cape-promoter.com). If this link is not accessible, please use [http://47.101.71.81](http://47.101.71.81). Please note that the runtime is positively correlated with the number of mutated sequences. Processing approximately 1 million sequences takes about 5 hours (we will send you an email about the results after the calculation is finished). We have provided an instruction guide in both Github and website to help you use the website. Given potential updates, refer to this repository for accessing the website. 


## Abstract
Predicting the strength of promoters and guiding their directed evolution is a crucial task in synthetic biology. This approach significantly reduces the experimental costs in conventional promoter engineering. Previous studies employing machine learning or deep learning methods have shown some success in this task, but their outcomes were not satisfactory enough, primarily due to the neglect of evolutionary information. 

In this paper, we introduce the Chaos-Attention net for Promoter Evolution (CAPE) to address the limitations of existing methods. We comprehensively extract evolutionary information within promoters using merged chaos game representation and process the overall information with modified DenseNet and Transformer structures. Our model achieves state-of-the-art results on two kinds of distinct tasks related to prokaryotic promoter strength prediction. The incorporation of evolutionary information enhances the model’s accuracy, with transfer learning further extending its adaptability. 

Furthermore, experimental results confirm CAPE’s efficacy in simulating _in silico_ directed evolution of promoters, marking a significant advancement in predictive modeling for prokaryotic promoter strength. Our paper also presents a user-friendly website for the practical implementation of _in silico_ directed evolution on promoters.

## Model Architecture
![The Model Architecture of CAPE](https://github.com/BobYHY/CAPE/blob/main/Figure1.png)
Here is the model architecture of CAPE. First, we employed Basic Local Alignment Search Tool (BLAST) and the Needleman-Wunsch (NW) algorithm to search for sequences exhibiting a certain level of similarity with the target promoter within a prokaryotic promoter database. 

Subsequently, we applied a novel method firstly introduced in this paper, referred to as merged CGR, to convert the promoter sequence into image data capturing evolutionary information. Alongside image information, we applied the kmer2vec method to extract textual information from the promoter sequences. The above two types of information will be input into two different deep learning networks, namely DenseNet and Transformer, respectively. We adapted the structure of DenseNet and Transformer to suit our tasks. 

The results processed by both models are fed into a fully connected network for integration. Finally, our model can output the predicted strengths of given promoter sequences. Moreover, we introduced a fine-tuning network for transfer learning, which enhances the model’s ability to adapt to various downstream tasks.

## Availability

### Structure for Folder CAPE_Final (Code for the Model)
- data: Contains all relevant data files.
- dataset: Contains codes for dataset loading and processing (including Merged CGR matrix generation).
- model: Contains the network model files.
- train_and_val: Contains training and evaluation related content.
- main_exp_task1.py: Code for Task 1.
- main_exp_task2.py: Code for Task 2.
- best_previous.pth: Pre-trained model for Task 2.

### Structure for Folder Website_server (Code for the Website)
- best_previous.pth: Trained model weights for the model. It is worth noticing that the Pearson correlation coefficient and other metrics reported in the article are computed via K-fold cross-validation and this weights do not correspond to the specific parameter configurations tied to the reported performance metrics.
- word2vec.npy: Embedding file containing vector representations of k-mers.
- promoter50.fasta & blast_promoter50.*: The promoter library file and files used for BLAST analysis.
- Run.py: Code for directed evolution.
- run.sh: Shell script to run the entire codebase.

### Reproducing Our Code
To reproduce our code, the following environments are required:
- MAC OS, Linux or Windows.
- Python 3.8+.
- PyTorch 2.0.0
- CUDA 11.7  if you need train deep learning model with gpu.
- numpy 1.24.1
- pandas 2.0.3
- Biopython 1.81
- scikit-learn 1.2.2
- scipy 1.10.1
- entmax 1.1
- einops 0.6.0

Then you can reproduce our codes for both Task 1 and Task 2 by

    ```
    cd CAPE_Final
    python main_exp_task1.py
    python main_exp_task2.py
    ```

### Directed Evolution for Your Sequences
For guidance on directed evolution for your sequence, please use our website [http://www.cape-promoter.com](http://www.cape-promoter.com). If this link is not accessible, please use [http://47.101.71.81](http://47.101.71.81). The runtime is positively correlated with the number of mutated sequences. Processing approximately 1 million sequences takes about 5 hours. We have provided an instruction guide in both Github and website to help you use the website. Given potential updates, refer to this repository for accessing the website. 

### Supplementary Information
For supplementary information, please refer to the Supplementary_Materials_All.pdf.
