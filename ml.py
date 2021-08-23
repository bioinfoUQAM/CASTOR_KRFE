# Imports
import data
import kmers
import joblib
import matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Function to instantiate a linear svm classifier
def svm():
	# Return a linear svm classifier
	return SVC(kernel = 'linear', C = 1, cache_size = 2000)

# Function to instantiate a random forest classifier
def randomForest():
	# Return a random forest classifier
	return RandomForestClassifier(n_estimators = 200, max_depth = None, random_state = 0, n_jobs = -1)

# Function to Compute the F1 score
def compute_f1_score(y, y_pred):
	# Return the computed F1 score
	return  f1_score(y, y_pred, average ="weighted")

# Function to transform features by scaling each feature between 0 and 1
def minMaxScaler(X):
	# Return scaled features
	return MinMaxScaler(feature_range = (0, 1), copy = False).fit_transform(X)

# Function to fit a model using a set of k-mers
def fit(parameters):
	# Get the parameters
	model_path = str(parameters["model_path"])
	# Get the path of the k-mers file
	k_mers_path = str(parameters["k_mers_path"])
	# Get the path of the training fasta file
	file_path = str(parameters["training_fasta"])
	# Load the training data
	D = data.loadData(file_path)
	# Get the set of k-mers
	K = kmers.loadKmers(k_mers_path)
	# Get the k-mers length
	k = len(list(K.keys())[0])
	# Generate the samples matrix (X) and the target values (y)
	X, y = matrix.generateSamplesTargets(D, K , k)
	#  Instantiate a linear svm classifier
	clf = svm()
	# Fit the classifier
	clf.fit(X, y)
	# Save the model
	joblib.dump(clf,  model_path)
	# Displays a confirmation message
	print("Model saved at the path:", model_path)

# Function to predict a set of sequences
def predict(parameters):
	# Get the path of the model file
	model_path = str(parameters["model_path"])
	# Get the  path of the k-mers file
	k_mers_path = str(parameters["k_mers_path"])
	# Get the testing fasta file
	file_path = str(parameters["testing_fasta"])
	# Get the prediction file path
	prediction_path = str(parameters["prediction_path"])
	# Get the evaluation mode
	evaluation_mode = str(parameters["evaluation_mode"])
	# Load the training data
	D = data.loadData(file_path)
	# Get the set of k-mers
	K = kmers.loadKmers(k_mers_path)
	# Get the k-mers length
	k = len(list(K.keys())[0])
	# Generate the samples matrix (X) and the target values (y)
	X, y = matrix.generateSamplesTargets(D, K , k)
	# Load the classifier
	clf = joblib.load(model_path)
	# Predict the sequences
	y_pred = clf.predict(X)
	# If evaluation mode is egal to True
	if evaluation_mode == "True":
		# If the target values list is empty
		if len(y) == 0: print("Evaluation cannot be performed because target values are not given")
		# Else display the classification report
		else: print("Classification report \n", classification_report(y, y_pred))
	# Save the predictions
	f = open(prediction_path, "w")
	# Write the header
	f.write("id,y_pred\n")
	# Iterate through the predictions
	for i, y in enumerate(y_pred): 
		# Save the current prediction
		f.write(D[i][0] + "," + y + "\n")
	# Close the file
	f.close()
	# Displays a confirmation message
	print("Predictions saved at the path:", prediction_path)
