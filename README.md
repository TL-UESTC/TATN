# [Temperature Adaptive Transfer Network for Cross-Domain State of Charge Estimation of Li-ion Batteries at Different Ambient Temperatures](https://ieeexplore.ieee.org/document/9944189)
Liyuan Shen, Jingjing Li, Jieyan Liu, Lei Zhu, Heng Tao Shen  

Abstract:State of charge (SOC) estimation plays an important role in battery management system (BMS), which serves to ensure the safety of batteries. Existing data-driven methods for SOC estimation of Li-ion batteries (LIBs) rely on massive labelled data and the assumption that training and testing data share the same distribution. However, in real-world, there is only unlabelled target data and these exists distribution discrepancy caused by external or internal factors such as varying ambient temperatures and battery aging, which makes existing methods invalid. Thus, it is necessary to develop effective unsupervised methods. To address the challenges, temperature adaptive transfer network (TATN) is proposed, which can mitigate domain shift adaptively by mapping data to high-dimensional feature spaces. TATN consists of pre-training stage and transfer stage. At pretraining stage, two-dimensional convolutional neural network (2D-CNN) and bidirectional long short-term memory (BiLSTM) are used for temporal feature extraction. At transfer stage, adversarial adaptation and maximum mean discrepancy (MMD) are utilized to minimize domain divergence. Furthermore, a novel label-selection method is proposed to select reliable pseudo labels. Extensive transfer experiments are performed. In pre-training stage, TATN achieves mean absolute error (MAE) and root mean square error (RMSE) of 0.294% and 0.366% for training and average errors of 1.09% and 1.44% for testing. In transfer stage, compared with other methods, TATN reduces average MAE and RMSE by 66% and 78% under semi-supervised scenario, 71% and 68% under unsupervised scenario, 52% and 42% at online testing. The results indicate TATN can achieve state-of-the-art performance in practical applications.
# Usage
* conda environment   
`conda env create -f env.yaml`
* Dataset  
more dataset for LIBs can be downloaded from [HERE](https://docs.google.com/spreadsheets/d/10w5yXdQtlQjTTS3BxPP233CiiBScIXecUp2OQuvJ_JI/edit#gid=0)
* Data processing  
put your data fold in normalized_data/  
then run this code  
`python normalized_data/dataprocess.py`      
* To pretrain a source model    
`python run.py --mode pretrain --mkdir [res fold] --source_data_path [] --source_temp [] --epochs --batch_size`   
(check run.py for more arguments)  
The model is saved in run/res fold/saved_model/best.pt  
* Pseudo label    
Use pre-trained source model to generate pseudo labels for target data:    
'python pseudo.py --temp --model --file`   
* To transfer a model  
`python run.py --mode train --mkdir [] --source_data_path --source_temp --target_data_path --target_temp --epochs --batch_size`  
(check run.py for more arguments)   
* To test a model  
`python run.py --mode test --mkdir [] --test_set [] --target_temp []`  
