# Import
import numpy
from joblib import Parallel, delayed

# Function to compute the occurence vector of sequence
def computeSequenceVector(d, K, k):
	# Generate an empty dictionary
	x = {}
	# Initialize the dictionary with targets as keys and 0 as value
	x = x.fromkeys(K.keys(), 0)
	# Go through the sequence 
	for i in range(0, len(d[1]) - k + 1, 1):
		# Try to increment the current k-mer value by 1
		try: x[d[1][i:i + k]] = x[d[1][i:i + k]] + 1
		# Pass if the k-mers does not exist
		except: pass
	# Return the vector and associated target
	return [list(x.values()), d[2]]


# Function to generate the samples matrix (X) and the target values (y)
def generateSamplesTargets(D, K, k):
	# Samples matrix
	X = []
	# Target values
	y = []
	# Iterate through the data
	data = Parallel(n_jobs = -1)(delayed(computeSequenceVector)(d, K, k) for d in D)
	# Add to the matrices
	for d in data: 
		X.append(d[0])
		y.append(d[1])

	# Convert to numpy array format
	X = numpy.matrix(X)
	y = numpy.array(y)
	# Return the samples matrix (X) and the target values (y)
	return X, y