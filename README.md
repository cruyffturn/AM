#Required packages

autograd                  1.4 
matplotlib                3.3.3
numpy                     1.19.5 
scipy                     1.5.3 
scikit-learn              1.0.2 
panda s(with excel read)  1.1.4 
tensorflow                2.4.0 
causallearn               0.1.3.1 
networkx                  2.7.1
joblib                   0.17.0

#List of helper files

-helper_bnlearn:    Reads the bnlearn data, selects the adversarial parameters
-helper_dag:        Subrotiines relavent to sampling or estimatating dags
-helper_data:       Sachs dataset and trivariate dataset helper
-helper_draw:       Plots the loss functions
                    
-helper_em_tf:      Implements the WEM, and other EM related functions in tf
-helper_mvn_tf:     Subroutines relavent to mvn distribution in tf
-helper_tf_model:   Custom Keras model for the Algorithm 2

-helper_rs:         Implements the local rejection sampling (RS) attack


-helper:            Calls modeler algorithms (missDAG, missPC, etc) and 
                    calculates graph distance
-helper_mvn:        Subroutines relavent to mvn distribution
-helper_em:         Implements missDAG

-train_bnlear_adv:  Trains LAMM for the bnlearn dataset
-train_sachs_adv:   Trains LAMM for the sachs dataset
-train_trivariate:  Trains LAMM for Gaussian SCM I


-load_bnlearn:      Tests the LAMM for bnlearn
-load_rs:           Trains & tests the local rs attack on Gaussian SCM II.
-load_sachs:        Tests the LAMM for sachs dataset
-load_trivariate:   Tests the LAMM for Gassian SCM I

-External.notears:  Contains the modified external notears package accessed at
                    https://github.com/xunzheng/notears

#Sachs dataset requires downloading the supplementary files
1. Download https://www.science.org/doi/suppl/10.1126/science.1105809/suppl_file/sachs.som.datasets.zip
2. Put "1. cd3cd28.xls" to the project directory.


#Steps for testing the LAMM model:

1. Call the training file ex: train_sachs_adv.py,
2. Training curve is saved in a new folder starting with train_model_cfg_4_xx,
3. Call the corresponding test file ex: load_sachs.py,
-You can specify the testing setting with parameters: bool_mcar and exp_type

4. It will output the results in the folder train_model_cfg_4_xx,
-2_denea.png, summary_exp_type_1.csv:   results for missPC, imputation
-init_denea.png, summary.csv:           results for missDAG

#Steps for testing the local RS attack:

1. Call load_rs.py
-You can specify the testing setting with parameters: bool_mcar and exp_type

2. It will output the results in the folder train_model_cfg_4_xx,
-2_denea.png, summary_exp_type_1.csv:   results for missPC, imputation
-init_denea.png, summary.csv:           results for missDAG
