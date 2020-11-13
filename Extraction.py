# Imports
import numpy
import K_mers
import Matrices
import Preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

# Function core of feature extraction
def extractKmers(T, training_data, k_min, k_max, features_min, features_max):
	# Contains scores lists of different lengths of k
	scores_list = []
	# Contains list of k-mer for each iteration of rfe
	supports_list = []
	# List of different lengths of k-mer
	k_mers_range = range(k_min, k_max + 1)
	# Classifier svm
	classifier = svm.SVC(kernel = 'linear', C = 1)

	# Perform the analysis for the different sizes of k
	for k in k_mers_range:
		# Start feature extraction based on k-mers of length k
		print("\nBeginning of the " + str(k) + "_mer(s) analysis")

		# Generarte list of k-mer
		print("Generate K-mers...")
		k_mers = K_mers.generate_K_mers(training_data, k)
		
		# Genrate matrice attributes and matrice class
		print("Generate matrices...")
		X, y = Matrices.generateMatrice(training_data, k_mers, k)
		y = numpy.asarray(y)

		# Apply MinMaxScaler (0, 1)
		X = Preprocessing.minMaxScaling(X)

		# If more than features_max  apply RFE (remove 10 % of features to remove at each iteration)
		if len(X[0]) > features_max:
			print("Preliminary recursive feature elimination...")	
			from sklearn.feature_selection import RFE
			rfe = RFE(estimator = classifier, n_features_to_select = features_max, step = 0.1)
			X = numpy.matrix(X)
			X = rfe.fit_transform(X, y)

			# Update list of k_mers
			for i, value in enumerate(rfe.support_):
				if value == False: k_mers[i] = None
			k_mers = list(filter(lambda a: a != None, k_mers))

		# Recursive feature elimination
		from RFE import RFE
		print("Recursive feature elimination...")
		rfe = RFE(estimator = classifier, n_features_to_select = 1, step = 1)
		rfe.fit(X,y) 
		
		# Scores and supports of the actual iteration
		scores = [] 
		supports = []
		
		# Evaluation
		for i, supports_rfe in enumerate(rfe.supports):
			# Variables
			temp_index = []
			temp_k_mers = []

			# Print percentage of advancement
			print("\rFeature subset evaluation :", round((i + 1) / len(rfe.supports) * 100, 0), "%", end = '')

			# Selects k-mers with support equal True
			for j, support in enumerate(supports_rfe):
				if rfe.supports[i][j] == True: temp_index.append(j)

			# Replace the support by the k-mers 
			for t in temp_index: temp_k_mers.append(k_mers[t])
			rfe.supports[i] = temp_k_mers

			# Evaluation method
			stratifiedKFold = StratifiedKFold(n_splits = 5, shuffle = False, random_state = None)
			y_pred = cross_val_predict(classifier, X[:,temp_index], y, cv = stratifiedKFold, n_jobs = 4)
			score = f1_score(y, y_pred, average = 'weighted')

			# Save score and features of this iteration 
			scores.append(score)
			supports.append(rfe.supports[i])

		# Save the list of scores and feature subsets for this length of k-mers 
		scores_list.append(scores)
		supports_list.append(supports)

	# Changes the order of the lists for the graphic 
	for i, e in enumerate(scores_list):
		scores_list[i].reverse()
		supports_list[i].reverse()

	# Identify solution
	print("\n\nIdentify optimal solution...")
	# Best score of the evaluations
	best_score = 0
	# Optimal score in relation with treshold
	optimal_score = 0
	# Best k-mer list
	extracted_k_mers = []
	# Best length of k
	identified_k_length = 0

	# Identify best solution
	for i, s in enumerate(scores_list):
		if max(s) > best_score:
			best_score = max(s)
			index = s.index(max(s))
			identified_k_length = k_mers_range[i]
			extracted_k_mers = supports_list[i][index]
		elif max(s) == best_score:
			if s.index(max(s)) < index:
				best_score = max(s)
				index = s.index(max(s))
				identified_k_length = k_mers_range[i]
				extracted_k_mers = supports_list[i][index]
		else: pass

	# Identify optimal solution
	for i, l in enumerate(scores_list):
		for j, s in enumerate(l):
			if s >=  best_score * T and j <= index: 
				optimal_score = s
				index = j
				identified_k_length = k_mers_range[i]
				extracted_k_mers = supports_list[i][index]
	if optimal_score == 0: optimal_score = best_score


	# Save plot results
	fig = plt.figure(figsize = (12, 10) )
	for i, s in enumerate(scores_list):
		label = str(k_mers_range[i]) + "-mers"
		plt.plot(range(len(s)), s, label = label)
	plt.ylabel("F-measure")
	plt.xlabel("Number of features")
	plt.axvline(index, linestyle = ':', color = 'r')
	title = "F-measure : " + str(round(optimal_score, 3)) + " K-mer size : " + str(identified_k_length) + " Number of features : " + str(index + 1)
	plt.title(title)
	plt.legend()
	fname = str("Output/Analysis.png")
	plt.savefig(fname)

	# Save extracted k-mers
	f = open("Output/Kmers.txt", "w")
	for i in extracted_k_mers: f.write(str(i) + "\n");
	f.close()

	# Return identified k-mers and their length
	return extracted_k_mers, identified_k_length
