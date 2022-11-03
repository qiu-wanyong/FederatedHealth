# FederatedHealth_HeartSound
The horizontal FL (HFL) and vertical FL (VFL) paradigms for heart sound analysis.

# Algorithm_Model
The source code and models for paper "Heart Sound Abnormality Detection from Multi-institutional Collaboration: Introducing a Federated Ensemble Learning Framework"

## Abstract
Early diagnosis of cardiovascular diseases is a crucial task in medical practice. With the application of computer audition in the healthcare field, artificial intelligence (AI) has been applied to clinical non-invasive intelligent auscultation of heart sounds to provide rapid and effective pre-screening. However, AI models generally require large amounts of data which may cause privacy issues. Unfortunately, it is difficult to collect large amounts of healthcare data from a single centre. In this study, we propose federated learning (FL) optimisation strategies for the practical application in multi-centre institutional heart sound databases. The horizontal federated learning is mainly employed to tackle the privacy problem by aligning the feature spaces of the participants without information leakage. In addition, techniques based on deep learning have poor interpretability due to their ``black-box" property, which limits the feasibility of AI in real medical data. To this end, vertical federated learning is utilised to address the issues of model interpretability and data scarcity. Experimental results demonstrate that, the proposed FL framework can achieve good performance for heart sound abnormality detection by taking the personal privacy protection into account. Moreover, the interpretability of the vertical FL model can be improved by using the federated feature space.

#### Index Terms— Computer audition, federated learning, heart sound, information security, model interpretability

![](/Graphical Abstract.jpg)

Fig. 1. Paradigms and workflows of horizontal and vertical federated learning (FL) on multi-institutional heart sound databases.
 
## Main contributions:
 * This study has developed optimisation schemes for  HFL and VFL to analyse the applications of FL in different healthcare conditions and to verify the effectiveness of the models on multi-centre heart sound databases.
 * In the HFL modelling, we propose a privacy-preserving feature ID-based security aggregation method. It has the advantage of solving the issue of aligning the feature space of federated participants in HFL.
 * The VFL model in this study is used to solve the issue of unlabelled data for some federated institutions. Further, we propose an approach to balance model interpretability and patient privacy for VFL using Shapley values.
 
## Results
 * Horizontally-Federated Learning vs Data-Centralised Learning
  
 ![](/figures/HFL_results.jpg)
 
 Fig. 2. Fig. (a)-(b) show the variation of the performance (in [%]) of the  HFL model with the number of trees. Fig. (c)-(d) shows the variation of  the performance (in [%]) of the HFL. model with tree depth.
 
Table 2. Optimal values for the depth and number of trees; summary of experimental results (in [%]).

| Model      | Acc         | Se        |    Sp    |   UF1     |    UAR    |
| -----      | -----       | ----      |----      |----       |
| XGBoost    |  68.4       | 69.1      |67.6      | 68.4      |   68.4    |
| Homogeneous-
SecureBoost  |  67.5       | 62.1      |72.8      | 67.4      |   67.5    |


The important parameters are set, e. g., learning rate=0.3, subsample feature rate=1.0, and other parameters to their default values.

 ![](/figures/matrix.jpg)
 
 Fig. 3. Normalised confusion matrix (in [%]) of the FL.
 
![](/figures/shap1.jpg)

(a) The contribution of significant auDeep features from all class predictions for the XGBoost model (average feature importance).
  
![](/figures/shap2.jpg)

(b) The contribution of significant auDeep features from all class predictions for the FL model (average feature importance).

Fig. 4. The plot sorts the features by the mean of Shapley values for all class predictions and uses the Shapley values to show the average impact on the model output magnitude of the features. Top 10 most impactful features are shown above.
  
## Availability
1. KSoF: Access to the data can be requested from the Kassel State of Fluency (KSoF) dataset at https://zenodo.org/record/6801844.

2. SHAP (SHapley Additive exPlanations) is a game-theoretic method to explain the output of ML models. https://shap.readthedocs.io.

3. FATE (Federated AI Technology Enabler) supports the FL architecture, as well as the secure computation and development of various ML algorithms. https://github.com/FederatedAI/FATE.

## References
[1] Bjoern Schuller, Anton Batliner, Shahin Amiriparian, Christian Bergler, Maurice Gerczuk, Natalie Holz, Pauline Larrouy-Maestri, Sebastien Bayerl, Korbinian  Riedhammer, Adria Mallol-Ragolta, et al., The ACM Multimedia 2022 Computational Paralinguistics Challenge: Vocalisations, Stuttering, Activity, &amp; Mosquitoes, in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 7120 7124.

[2] Sebastian P Bayerl, Alexander Wolff von Gudenberg, Florian H onig, Elmar N oth, and Korbinian Riedhammer, KSoF: The Kassel State of Fluency Dataset A Therapy Centered Dataset of Stuttering, arXiv preprint arXiv:2203.05383, pp. 1 8, 2022.

[3] Kun Qian, Zixing Zhang, Yoshiharu Yamamoto, and Bjoern W Schuller, Artificial Intelligence Internet of Things for the Elderly: From Assisted Living to Healthcare  Monitoring, IEEE Signal Processing Magazine, vol. 38, no. 4, pp. 78 88, 2021.

[4] Wanyong Qiu, Kun Qian, Zhihua Wang, Yi Chang, Zhihao Bao, Bin Hu, Bjoern W Schuller, and Yoshiharu Yamamoto, A Federated Learning Paradigm for Heart Sound Classification, in Proceedings of the Engineering in Medicine &amp; Biology Society (EMBC). IEEE, 2022, pp. 1045 1048.

## Cite As
Yongzi Yu, Wanyong Qiu, Chen Quan, Kun Qian, Zhihua Wang, Yu Ma, Bin Hu∗, Bjoern W. Schuller and Yoshiharu Yamamoto, “Federated intelligent terminals facilitate stuttering monitoring”, in Proceedings of ICASSP, pp. 1-5, Submitted, October 2022.


