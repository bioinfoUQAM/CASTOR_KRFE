###############
### IMPORTS ###
############### 
import Data
import Extraction
import Evaluation

####################
### INFORMATIONS ###
####################
print("*******************")
print("*** CASTOR-KRFE ***")
print("*******************\n")

print("Alignment-free method to extract discriminant genomic subsequences within pathogen sequences.\n")

#################
### VARIABLES ###
#################

# Threshold (percentage of performance loss in terms of F-measure to reduce the number of attributes)
T = 0.999
# Minimum length of k-mer(s)
k_min = 1
# Maximum length of k-mer(s)
k_max = 5
# Minimum number of features to identify
features_min = 1
# Maximum number of features to identify
features_max = 100
# Training fasta file path
training_fasta = "Input/HIVGRPCG/data.fasta"
# Training fasta file path
training_csv = "Input/HIVGRPCG/target.csv"
# Testing fasta file path
testing_fasta = "Input/HIVGRPCG/data.fasta"
# Testing fasta file path
testing_csv = "Input/HIVGRPCG/target.csv"

###########################
### LOAD TRAINING DATA  ###
###########################
print("\nLoading of the training dataset...")
if Data.checkTrainFile(training_fasta, training_csv) == True: training_data = Data.generateTrainData(training_fasta, training_csv)

############################
### FEATURES EXTRACTION  ###
############################
print("\nStart feature extraction...")
extracted_k_mers, identified_k_length = Extraction.extractKmers(T, training_data, k_min, k_max, features_min, features_max)

########################
### MODEL EVALUATION ###
########################
print("\nEvaluation of the prediction model...")
Evaluation.cross_validation(training_data, extracted_k_mers, identified_k_length, training_data)

###########################
### LOAD TESTING DATA  ###
###########################
print("\nLoading of the testing dataset...")
if Data.checkTestFile(testing_fasta, testing_csv) == True: testing_data = Data.generateTestData(testing_fasta, testing_csv)

###################
### PREDICTION  ###
###################
if len(testing_data[0]) == 2: 
	print("\nPrediction without evaluation...")
	Evaluation.prediction(training_data, testing_data, extracted_k_mers, identified_k_length)
else: 
	print("\nPrediction with evaluation...")
	Evaluation.predictionEvaluation(training_data, testing_data, extracted_k_mers, identified_k_length)

###########
### END ###
###########
print("\nEnd of the program ")

