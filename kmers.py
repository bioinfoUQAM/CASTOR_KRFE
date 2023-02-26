# Import
import re
import Bio.SeqIO
from collections import defaultdict
from joblib import Parallel, delayed

# Define a function to count the occurrences of k-mers in a single sequence
def count_seq(seq, k):
    # Create an empty dictionary to store the counts for each k-mer
    counts = defaultdict(int)
    # Iterate through the sequence, considering k-mer windows of length k
    for j in range(len(seq) - k + 1):
        # Extract the k-mer from the sequence
        kmer = seq[j:j+k]
        # Increment the count for this k-mer
        counts[kmer] += 1
    # Return the dictionary of k-mer counts
    return counts

# Define a function to get the k-mers from a list of sequences
def getKmers(D, k):
    # Use the Parallel function to parallelize the counting of k-mers
    # This will speed up the computation on multi-core CPUs
    counts = Parallel(n_jobs=-1)(delayed(count_seq)(d[1], k) for d in D)
    # Merge the counts into a single dictionary, with 0 as the default value
    all_counts = defaultdict(int)
    for count in counts:
        all_counts.update(count)
    # Return the dictionary of all k-mer counts
    return all_counts

# Function to save the extracted k-mers
def saveExtractedKmers(k_mers_path, k_mers):
	# Open file
	file = open(k_mers_path, "w")
	# Iterate through the k-mers
	for i, k in enumerate(k_mers):
		# Save the current k-mer
		file.write(">" + str(i) + "\n" + k + "\n")
	# Close the file
	file.close()

# Function to load the set of k-mers from a fasta file
def loadKmers(k_mers_path):
	# Initialize an empty dictionary for the k-mers
	K = {}
	# Iterate through the k-mers
	for record in Bio.SeqIO.parse(k_mers_path, "fasta"):
		# Save the current k-mer
		K[str(record.seq.upper())] = 0
	# Return the dictionary of k-mers
	return K