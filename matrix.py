# Import
import numpy

# Function to generate the samples matrix (X) and the target values (y)
def generateSamplesTargets(D, K, k):
	# Samples matrix
	X = []
	# Target values
	y = []
	# Iterate through the data
	for d in D:
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
		# Save the number of occurrences of k-mers
		X.append(list(x.values()))
		# Target the value
		try: y.append(d[2])
		# Pass if the target the value
		except: pass
	# Convert to numpy array format
	X = numpy.matrix(X)
	y = numpy.array(y)
	# Return the samples matrix (X) and the target values (y)
	return X, y
