import csv
import os

import pandas as pd
import statistics
import subprocess
from functions import preprocessing, PCA_function,sampling,ML_model


# Three Possible projects: P1: PRJNA312948   P2: PRJEB23709  P3: PRJNA356761
# Read in the data
exp = pd.read_csv('Data/all_tpm.txt', delimiter='\t',index_col=0)
clinic = pd.read_csv('Data/Melanoma pretreatment only clinic.csv', delimiter=',',index_col='Sample ID')
proportion = pd.read_csv('Data/all_proportion.txt', delimiter='\t',index_col=0)
exp_Count = pd.read_csv('Data/all_of_them_count.txt', delimiter='\t',index_col=0)

# Specifiy which data need to use
Used_projects =[]
P = 1
print("Specifiy the Projects used in the workflow (can be > 1).")
print("Stop adding with entering nothing")
while P:
    P = input("Provide the Project ENA IDs from (P1: PRJNA312948   P2: PRJEB23709  P3: PRJNA356761) : ")
    if P:
        Used_projects.append(P)
if len(Used_projects) < 1:
    print("At least 1 project should be used in this workflow!")
    exit()
Used_samples = [i for i in clinic.index if clinic.loc[i,'ENA project ID'] in Used_projects]
exp = exp[Used_samples]
exp_Count = exp_Count[Used_samples]
clinic= clinic.loc[Used_samples,:]

# Data_preprocessing steps on Gene expression data
exp_ProjectX_sd = preprocessing.preProcessing_Steps(exp, pre_processing_order=["log2","Standardiaztion","QN"])
print("\n")

# Plot the PCA plot by given feature from clinic
do_PCA = input("Run PCA analysis?(y/n) ")
print("\n")
if do_PCA == "y":
    col_for_PCA = input("Select which column to make a 2d PCA plot? ")
    print("Making PCA plot on {}".format(col_for_PCA))
    print("\n")
    PCA_function.PCA_plot(exp_ProjectX_sd,clinic,col_for_PCA)


# Remove genes with median TPM < 5 in both Responders and Non-repsonders
R = clinic.loc[clinic['Response']==1,:].index
NR = clinic.loc[clinic['Response']==0,:].index
low_TPM = [i for i in exp.index if statistics.median(exp.loc[i,R])<=5 and statistics.median(exp.loc[i,NR])<=5]
exp_Count = exp_Count.drop(low_TPM, axis=0)

# Export Count data and clinic batch to R script
exp_Count.to_csv("R_script/_count_data.csv")
clinic_batch = clinic[["Response"]]
clinic_batch["batch"] = [Used_projects.index(str(i))+1 for i in clinic['ENA project ID']]
clinic_batch.to_csv("R_script/_clinic_batch.csv")

# Split into 30% tesing datasets in Projects levels
output_train = pd.DataFrame()
output_test = pd.DataFrame()
repeat_time = int(input("Input the times of repeats (for testing code using 3): "))
# Pass the times of repeats to the R script
_times = pd.DataFrame()
_times['Times'] = [repeat_time]
_times.to_csv("R_script/_times.csv")
print("\n")
for i in range(700,700+3*repeat_time,3):
    temp_train_samples,temp_test_samples = [],[]
    for project_id in Used_projects:
        temp_row = [j for j in clinic.index if clinic.loc[j,'ENA project ID'] == project_id]
        a_train, a_test = sampling.split_testing_dataset(clinic.loc[temp_row,:],0.3)
        temp_train_samples = list(set(temp_train_samples+a_train))
        temp_test_samples = list(set(temp_test_samples+a_test))
    # Save the Samples ID in training and testing in each time
    output_train["X"+str(i)], output_test["X"+str(i)] = temp_train_samples, temp_test_samples
output_train.to_csv("R_script/_training_samples_inRepeats.csv")
output_test.to_csv("R_script/_testing_samples_inRepeats.csv")


# DEGs(differential expression gene analysis) in R
print("EdgeRs-DEGs start")
current_path = os.getcwd() + '/R_script'
path_to_Rscript = input('Plz input the path to the Rscript: ')
######### My Rscript path: /Library/Frameworks/R.framework/Resources/bin/Rscript
R_codes = subprocess.call('{} --vanilla {}/EdgeR_DEGs.r'.format(path_to_Rscript,current_path),
                         shell=True)
print("EdgeRs-DEGs for all repeats are done.\n")


# Generate a output_file to store the model performance under various number of features
output_file = "output/" + input("The output file name: ")
with open('{}.csv'.format(output_file), 'w') as f:
    write = csv.writer(f, delimiter=",")
    write.writerow(["Penalty_values", 'Predictive_on_test_datasets',
                    "all Accuracy means", "all Accuracy sd", "0 Accuracy means", "0 Accuracy sd",
                    "1 Accuracy means", "1 Accuracy sd", "Numbers of features", "Names of features"])
    f.close()


# Provide the list of penalty values used for LASSO regression
penalty_ls = [*list(ML_model.float_range(0.00001, 0.0001, '0.00001',5)), *list(ML_model.float_range(0.0001, 0.001, '0.0001',4)),
              *list(ML_model.float_range(0.001, 0.01, '0.001',3)), *list(ML_model.float_range(0.01, 0.1, '0.01',2)),
              *list(ML_model.float_range(0.2, 1, '0.1',1))]

# Run multiple times of repeats
num = 1
for i in range(700,700+3*repeat_time,3):
    print("Model Training Repeat: {} times".format(num))
    num+=1

    repeat = "X"+str(i)
    train_samples = output_train[repeat].to_list()
    test_samples = output_test[repeat].to_list()

    # Find the gene with absolute LogFC>=1
    DEG_ls = pd.read_csv("R_script/DEGs in repeats/{}_edgeR_DEGs.csv".format(repeat),index_col=0)
    DEG_ls = DEG_ls.loc[abs(DEG_ls['logFC'])>=1,:].index.tolist()
    print("{} genes with |logFC| between Responders & Non-responders >= 1".format(len(DEG_ls)))

    # Merge Clinic proportions and exression tables together
    data_TPM_Project = pd.concat([clinic[['Response', 'Gender(M0 F1)', 'Age']], proportion, exp_ProjectX_sd.loc[DEG_ls,:].T], axis=1, join='inner')
    data_TPM_Project_train = data_TPM_Project.loc[train_samples,:]
    data_TPM_Project_test = data_TPM_Project.loc[test_samples,:]

    # balance Responders and Non-responders in training datasets based on the project's level
    for ind in range(0,len(Used_projects)):
        if ind == 0:
            col_project = [i for i in train_samples if clinic.loc[i, 'ENA project ID'] == Used_projects[ind]]
            balanced_train = sampling.balance_subclass(data_TPM_Project_train.loc[col_project, :])
        else:
            col_project = [i for i in train_samples if clinic.loc[i, 'ENA project ID'] == Used_projects[ind]]
            temp_balanced = sampling.balance_subclass(data_TPM_Project_train.loc[col_project, :])
            balanced_train = pd.concat([balanced_train, temp_balanced], axis=0, join='inner')

    # Finally test the model
    y_LASSO = data_TPM_Project_train.loc[:, "Response"]
    X_LASSO = data_TPM_Project_train.drop("Response", axis=1)
    ML_model.ML_predictor_building(X_lasso=X_LASSO, y_lasso=y_LASSO, data_train_balanced=balanced_train,
                                   penality_list=penalty_ls, file=output_file, Model='SVM',
                                   data_validation=[data_TPM_Project_test])


## Delete temp files
import os, glob
for filename in glob.glob("R_script/_*"):
    os.remove(filename)