# DeepSSC
# A biphasic Deep Semi-supervised framework for Subtype Classification and biomarker discovery

---
This repository contains source code and original/preprocessed datasets for the paper "A biphasic Deep Semi-supervised framework for Suptype Classification and biomarker discovery". 
#### 1. Introduction
---
To take full advantage of the unprecedented development of -omics technologies and generate further biological insights into human disease, it is a pressing need to develop novel computational methods for integrative analysis of multi-omics data. Here, we proposed a biphasic Deep Semi-supervised multi-omics integration framework for Subtype Classification and biomarker discovery, DeepSSC. In phase 1, each denoising autoencoder was used to extract a compact representation for each -omics data, and then they were concatenated and put into a feed-forward neural network for subtype classification. In phase 2, our Biomarker Gene Identification (BGI) procedure leveraged that neural network classifier to render subtype-specific important biomarkers. We also validated our given results on independent dataset. We demonstrated that DeepSSC exhibited better performance over other state-of-the-art techniques concerning classification tasks. As a result, DeepSSC successfully detected well-known biomarkers and hinted at novel candidates from different -omics data types related to the investigated biomedical problems. 

#### 2. Analysis Pipeline
---
![Figure1](https://i.imgur.com/mfOUzV9.png)
** Figure 1 | Framework of DeepSSC. ** In phase 1, the two preprocessed profiles (GE and CNA) from TCGA were fed into their own denoising autoencoders to generate two independent representations, zGE and zCNA . Then, zGE and zCNA  were concatenated into a single input layer attached with two FC layers to create a neural network classifier. Finally, this neural network classifier became the input of our BGI procedure in phase 2 to detect candidate biomarkers. To demonstrate the efficacy of DeepSSC, those biomarkers should help classify cancer patients into different subtypes well by only using simple machine learning models. Besides, those biomarkers also were further investigated in terms of their biological functions. Note that labeled and unlabeled data were defined clearly at the ‘Materials and Methods’ section in the work. Abbreviation: GE, mRNA expression; CNA, copy-number alterations; FC, fully-connected layer.

#### 3. Contact
---
Feel free to contact [Quang-Huy Nguyen](https://github.com/huynguyen250896) (huynguyen96.dnu@gmail.com) or [Duc-Hau Le](https://github.com/hauldhut) (hauldhut@gmail.com) for any questions about the paper, datasets, code and results.


