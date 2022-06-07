# Import
import Bio.SeqIO

# Function to load test data from a fasta file
def loadData(file_path):
	# Initialize the data matrix
	D = []
	# Iterate through the fasta file
	for record in Bio.SeqIO.parse(file_path, "fasta"):
		# If there is a class label, save id, sequence and class label in the data list
		try: 
			indexes = [i for i, c in enumerate(record.description) if c == "|"]
			D.append([record.description, str(record.seq.upper()).replace('N',''), record.description[indexes[len(indexes)-1] +1 :]])
		# If there is a no class label, save id and sequence in the data list
		except: D.append([record.descrition, str(record.seq.upper())])
	# Return the data matrix
	return D

# Function to load test data from a fasta file
def loadReferenceSequence(D, reference_sequence_file):
	# Iterate through the genbank file
	for gb_record in  Bio.SeqIO.parse(open(reference_sequence_file,"r"), "genbank"):
		D = [[gb_record.annotations["accessions"][0], str(gb_record.seq.upper()).replace('N',''), "Reference_sequence"]] + D
	# Return the data matrix
	return D