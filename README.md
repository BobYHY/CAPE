# CAPE: a deep learning framework with Chaos-Attention net for Promoter Evolution

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

### Structure for Folder CAPE_Final
- data: Contains all relevant data files.
- dataset: Contains dataset loading and processing (including Merged CGR matrix generation).
- model: Contains the network model files.
- train_and_val: Contains training and evaluation related content.
- main_exp_task1.py: Code for Task 1.
- main_exp_task2.py: Code for Task 2.
- best_previous: Pre-trained model for Task 2.

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
For guidance on directed evolution for your sequence, please use our website. We have provided an instruction guide to help you use the website. Given potential updates, refer to this repository for accessing the website. The website will be released once our article is published.

### Supplementary Information
For supplementary information, please refer to the Supplementary_Materials_All.pdf.
