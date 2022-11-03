# FederatedHealth_HeartSound
The horizontal FL (HFL) and vertical FL (VFL) paradigms for heart sound analysis.

# Algorithm_Model
The source code and models for paper "Heart Sound Abnormality Detection from Multi-institutional Collaboration: Introducing a Federated Ensemble Learning Framework"

## Abstract
Early diagnosis of cardiovascular diseases is a crucial task in medical practice. With the application of computer audition in the healthcare field, artificial intelligence (AI) has been applied to clinical non-invasive intelligent auscultation of heart sounds to provide rapid and effective pre-screening. However, AI models generally require large amounts of data which may cause privacy issues. Unfortunately, it is difficult to collect large amounts of healthcare data from a single centre. In this study, we propose federated learning (FL) optimisation strategies for the practical application in multi-centre institutional heart sound databases. The horizontal federated learning is mainly employed to tackle the privacy problem by aligning the feature spaces of the participants without information leakage. In addition, techniques based on deep learning have poor interpretability due to their ``black-box" property, which limits the feasibility of AI in real medical data. To this end, vertical federated learning is utilised to address the issues of model interpretability and data scarcity. Experimental results demonstrate that, the proposed FL framework can achieve good performance for heart sound abnormality detection by taking the personal privacy protection into account. Moreover, the interpretability of the vertical FL model can be improved by using the federated feature space.

#### Index Terms— Computer audition, federated learning, heart sound, information security, model interpretability

![](/figures/Graphical Abstract.jpg)
 
Fig. 1. Paradigms and workflows of horizontal and vertical federated learning (FL) on multi-institutional heart sound databases.
 
## Main contributions:
 * This study has developed optimisation schemes for  HFL and VFL to analyse the applications of FL in different healthcare conditions and to verify the effectiveness of the models on multi-centre heart sound databases.
 * In the HFL modelling, we propose a privacy-preserving feature ID-based security aggregation method. It has the advantage of solving the issue of aligning the feature space of federated participants in HFL.
 * The VFL model in this study is used to solve the issue of unlabelled data for some federated institutions. Further, we propose an approach to balance model interpretability and patient privacy for VFL using Shapley values.
 
## Results
 * Horizontally-Federated Learning vs Data-Centralised Learning
  
 ![](/figures/HFL_results.jpg)
 
 Fig. 4. Fig. (a)-(b) show the variation of the performance (in [%]) of the  HFL model with the number of trees. Fig. (c)-(d) shows the variation of  the performance (in [%]) of the HFL. model with tree depth.
 
Table 2. A SUMMARY OF RESULTS (IN [%]) FOR CLASSIC XGBOOST AND THE  HFL MODEL WITH OPTIMAL PARAMETERS.

|            | Acc         | Se        |    Sp    |   UF1     |    UAR    |
| -----      | -----       | ----      |----      |----       |----       |
| XGBoost    |  68.4       | 69.1      |67.6      | 68.4      |   68.4    |
| Homogeneous-SecureBoost  |  67.5     | 62.1     |72.8       | 67.4      |   67.5    |

Important parameters settings for the HFL and the XGBoost: tree depth=3,  tree number=30, subsample feature rate=1.0, learning rate=0.3.

 ![](/figures/HFL_matrix.jpg)
 
 Fig. 3. Normalised confusion matrix (in [%]) for the XGBoost and  Horizontal-SecureBoost models.
 
 
 Table 2. COMPARISON OF THE RESULTS (IN [%]) OF THE CONVENTIONAL  XGBOOST AND HETEROGENEOUS SECUREBOOST MODELS ON DATA  FOR EACH INSTITUTION.

|            | XGBoost(Centralised Data)        |Heterogeneous-SecureBoost       |
|            | Acc       | Se      |    Sp   |  UAR    | Acc     | Se      |    Sp   |   UAR  |
| -----      | -----     | ----    |----     |----     |----     |-----    | ----    |----    |
| Db         |  86.7     |85.2     |88.3     |86.8     |82.7     |82.0     |83.5     |82.7    |
| Dc         |  86.7     |85.7     |87.5     |86.6     |93.3     |85.7     |92.0     |92.9    |
| Dd         |  93.3     |87.5     |92.0     |93.8     |96.2     |89.6     |96.4     |97.2    |
| De         |  88.2     |85.6     |86.7     |87.8     |87.6     |82.6     |86.3     |84.3    |
| Df         |  86.8     |90.9     |81.3     |86.1     |79.5     |75.5     |71.3     |78.4    |


![](/figures/VFL_matrix.jpg)
 
Fig. 5. Normalised confusion matrix (in [%]) for XGBoost and Vertically-  SecureBoost models trained at D e database.

![](/figures/MMD.jpg)
 
Fig. 6. Normalised confusion matrix (in [%]) for XGBoost and Vertically-  SecureBoost models trained at D e database.

![](/figures/shap1.jpg)  

Fig.7 (a) and (c) show the summary bee-swarm plot of feature importance for the testing set of institutional database Db (Shapley values).  The plot sorts the features by the sum of Shapley value magnitudes for the abnormal class samples, and uses the Shapley values to show the  distribution of the impacts of each feature on the model output. The colour represents the feature value (red high, blue low). This reveals, for example,  that a higher Shapley value for MFCC (mfcc sam[1] quartile2 numeric) reduces the performance of abnormal predictions. Take the absolute mean  of Shapley values for each feature as the importance of that feature. Fig.7 (b) shows the average feature importance bar chart for the predictions of  the abnormal class, and Fig.7 (d) shows the results for all class predictions. (Note: To observe the change in importance of the federated features,  the feature ordering in Fig.7 (a) and (b) corresponds to each other, as do Fig.7 (c) and (d).)
  
![](/figures/shap2.jpg)

(b) The contribution of significant auDeep features from all class predictions for the FL model (average feature importance).

Fig. 8. Waterfall plots can provide us with the interpretability of a single prediction, and we can observe how features affect the prediction of an  abnormal sample. The horizontal axis is the Shapley value and the vertical axis is the value taken for each feature of that sample. Blue means that  the feature has a negative effect on the prediction, and the left arrow indicates a decrease in Shapley value. Red means that the feature has a  positive effect on the prediction, and the right arrow indicates an increase in Shapley value. As shown in Fig.8 (a), E[f(x)] is the baseline value  of SHAP and Feature 1524 = 32.358 produces a negative impact of 0.5. Cumulatively, until we reach the current model output f(x) = -0.248. (An  example of abnormal sample  a0169.wav ).
  
## Availability

1. Classification of Heart Sound Recordings (PhysioNet/CinC challenge): https://physionet.org/content/challenge-2016/1.0.0.

2. SHAP (SHapley Additive exPlanations) is a game-theoretic method to explain the output of ML models. https://shap.readthedocs.io.

3. FATE (Federated AI Technology Enabler) supports the FL architecture, as well as the secure computation and development of various ML algorithms. https://github.com/FederatedAI/FATE.

## References
[1] Bjoern Schuller, Anton Batliner, Shahin Amiriparian, Christian Bergler, Maurice Gerczuk, Natalie Holz, Pauline Larrouy-Maestri, Sebastien Bayerl, Korbinian  Riedhammer, Adria Mallol-Ragolta, et al., The ACM Multimedia 2022 Computational Paralinguistics Challenge: Vocalisations, Stuttering, Activity, &amp; Mosquitoes, in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 7120 7124.

[2] Sebastian P Bayerl, Alexander Wolff von Gudenberg, Florian H onig, Elmar N oth, and Korbinian Riedhammer, KSoF: The Kassel State of Fluency Dataset A Therapy Centered Dataset of Stuttering, arXiv preprint arXiv:2203.05383, pp. 1 8, 2022.

[3] Kun Qian, Zixing Zhang, Yoshiharu Yamamoto, and Bjoern W Schuller, Artificial Intelligence Internet of Things for the Elderly: From Assisted Living to Healthcare  Monitoring, IEEE Signal Processing Magazine, vol. 38, no. 4, pp. 78 88, 2021.

[4] Wanyong Qiu, Kun Qian, Zhihua Wang, Yi Chang, Zhihao Bao, Bin Hu, Bjoern W Schuller, and Yoshiharu Yamamoto, A Federated Learning Paradigm for Heart Sound Classification, in Proceedings of the Engineering in Medicine &amp; Biology Society (EMBC). IEEE, 2022, pp. 1045 1048.

## Cite As
Wanyong Qiu, Chen Quan, Lixian Zhu, Yongzi Yu, Zhihua Wang, Yu Ma, Mengkai Sun, Yi Chang, Kun Qian*, Bin Hu∗, Yoshiharu Yamamoto and Bjoern W. Schuller, “Heart Sound Abnormality Detection from Multi-institutional Collaboration: Introducing a Federated Ensemble Learning Framework”, JBHI, pp. 1-11, Submitted, October 2022.


