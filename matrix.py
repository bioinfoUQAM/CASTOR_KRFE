import numpy
from joblib import Parallel, delayed
from collections import defaultdict

# Function to compute the count of each k-mer in a sequence
def computeSequenceVector(d, k):
    sequence = d[1]
    vector = defaultdict(int)

    for j in range(len(sequence) - k + 1):
        kmer = sequence[j:j+k]
        vector[kmer] += 1
    
    return [d, vector]

# Function to generate the feature matrix X and target vector y from a list of sequences D, a dictionary of k-mers K, and a k-mer length k
def generateSamplesTargets(D, K, k):

    # Compute the k-mer counts for each sequence in parallel using joblib
    data_with_sequence_vector = Parallel(n_jobs=-1)(delayed(computeSequenceVector)(d, k) for d in D)
    
    # Extract the feature matrix X and target vector y from the sequence vectors and the original data
    X = []
    y = []

    for d in data_with_sequence_vector:
        # Create a list of k-mer counts with 0 for missing k-mers
        sequence_vector = [d[1][kmer] for kmer in K.keys()]
        X.append(sequence_vector)
        y.append(d[0][2])

    # Convert X and y to numpy arrays for use with scikit-learn
    X = numpy.array(X)
    y = numpy.array(y)

    return X, y