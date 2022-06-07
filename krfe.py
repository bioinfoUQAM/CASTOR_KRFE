# Imports
import ml
import sys
import data
import kmers
import numpy
import matrix
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Function to extract the discriminating k-mers
def extract(parameters):
	history = []
	# Initialize the maximum score
	max_score = 0
	# Initialize the best identified score
	best_score = 0
	# Initialize the best set of k-mers score
	best_k_mers = []
	# Initialize the best length of k
	best_k_length = 0
	# Initialize the best number of features
	best_features_number = 0

	# Get the parameters
	T = float(parameters["T"])
	k_min = int(parameters["k_min"])
	k_max = int(parameters["k_max"])
	k_mers_path = str(parameters["k_mers_path"])
	file_path = str(parameters["training_fasta"])

	# Load the training data
	D = data.loadData(file_path)
	# If the target values are not given
	if len(D[0]) != 3: 
		print("Target values are not given")
		sys.exit(0)

	# Iterate through the length of k-mers to explore
	for k in range(k_min, k_max + 1):
		# Displays the current analysis
		print("Analysis of the " + str(k) + "-mers")
		# Get the k-mers existing in the sequences
		K = kmers.getKmers(k, D)
		# Generate the samples matrix (X) and the target values (y)
		X, y = matrix.generateSamplesTargets(D, K , k)
		# Scale the features between 0 and 1
		X = ml.minMaxScaler(X)
		# If it is possible to apply a variance filter
		try:
			# Instancies the filter method
			varianceThreshold = VarianceThreshold(threshold = 0.01)
			# Apply the filter
			X = varianceThreshold.fit_transform(X)
			# Compute the list of k-mers indices to retain 
			indices = [i for i, value in enumerate(varianceThreshold.get_support()) if value == True]
			# Update the list of k-mers
			K = dict.fromkeys(list(itemgetter(*indices)(list(K.keys()))), 0)
			# Clear the indices list
			indices.clear()
		# If not, pass on
		except: pass

		# Instantiate a linear svm classifier
		clf = ml.svm()
		# Preliminary RFE if n features > 1000 
		rfe = RFE(estimator = clf , n_features_to_select = 1000, step = 0.1)
		# Fit and transform the initial matrix
		X = rfe.fit_transform(X, y)
		# Compute the list of k-mers indices to retain 
		indices = [i for i, value in enumerate(rfe.support_) if value == True]
		# Update the list of k-mers
		K = dict.fromkeys(list(itemgetter(*indices)(list(K.keys()))), 0)
		# Clear the indices list
		indices.clear()

		# List of scores related to each subset of features
		scores = []
		# List of number features related to each subset of features
		n_features = []
		# List of indices related to each subset of features
		selected_features = []
		# Initialize the empty list of indices
		indices = numpy.empty(0, int)
		# Apply recursive feature elimination
		rfe = RFE(estimator = clf , n_features_to_select = 1, step = 1).fit(X, y)
		# Iterate through the subset of features
		for i in range(1, rfe.ranking_.max()):
			# Merge the indices of this iteration to the list of indices
			indices = numpy.concatenate((indices,  numpy.where(rfe.ranking_ == i)[0]), axis = None)
			# Save the indices related to the actual subset of features
			selected_features.append(indices)
			# Split the data using stratified K-Folds cross-validator
			skf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
			# Perform the cross-validation for the actual subset of features
			y_pred = cross_val_predict(clf, X[:,indices], y, cv = skf, n_jobs = -1)
			# Compute the F1 score of the actual subset of features
			score = ml.compute_f1_score(y, y_pred)
			# Save the score of the actual subset of features
			scores.append(score)
			# Save the number of features of the actual subset of features
			n_features.append(len(indices))

		# Compute evaluation and save results for all features
		skf = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = True)
		y_pred = cross_val_predict(clf, X, y, cv = skf, n_jobs = -1)
		score = ml.compute_f1_score(y, y_pred)
		scores.append(score)
		n_features.append(X.shape[1])
		selected_features.append(numpy.where(rfe.ranking_ != 0)[0])

		# Get the best solution for this length of k-mers (Better score with lesser number of features)
		if max(scores) > max_score:
			max_score = max(scores)
			best_score = max_score
			best_k_length = k
			best_features_number = n_features[scores.index(max(scores))]
			best_k_mers = [list(K.keys())[i] for i in selected_features[ n_features.index(best_features_number)]]
			
		# Get best solution according to the threshold T
		for s, n in zip(scores, n_features):
			if s >= max_score*T and n <= best_features_number: 
				best_score = s
				best_k_length = k
				best_features_number = n
				best_k_mers = [list(K.keys())[i] for i in selected_features[n_features.index(best_features_number)]]

		# Save the history
		history.append(scores)

	# Plot the history
	for i, h in enumerate(history): 
		label = str(list(range(k_min, k_max + 1))[i]) + "-mers"
		plt.plot(list(range(1, len(h) + 1))[0:100], h[0:100], label = label)
		plt.axvline(best_features_number, linestyle=':', color='r')
		plt.suptitle("Distribution of F1-scores according to the length of k and the number of features", fontsize = 12)
		plt.title("Solution: F1 score = " + str(round(best_score, 2)) + ", Number of features = " + str(best_features_number) + ", Length of k = " + str(best_k_length), fontsize = 10)
		plt.xlabel('Number of features', fontsize = 10)
		plt.ylabel('F1 score', fontsize = 10)
		plt.legend()
	plt.show()

	# Save the extracted k-mers
	kmers.saveExtractedKmers(k_mers_path, best_k_mers)

	# Dipslay the solution
	print("\nIdentified solution:")
	print("Evaluation score (F1 score) =", best_score)
	print("Length of k =", best_k_length)
	print("Number of k-mers =", best_features_number)
	print("Extracted k-mers =", best_k_mers)
	print("Extracted k-mers saved at the path:", k_mers_path)