# Import
import Bio.SeqIO

# Function to load test data from a fasta file
def loadData(file_path):
	# Initialize the data matrix
	D = []
	# Iterate through the fasta file
	for record in Bio.SeqIO.parse(file_path, "fasta"):
		# If there is a class label, save id, sequence and class label in the data list
		try: D.append([record.id[0:record.id.index("|")], str(record.seq.upper()), record.id[record.id.index("|") + 1:]])
		# If there is a no class label, save id and sequence in the data list
		except: D.append([record.id, str(record.seq.upper())])
	# Return the data matrix
	return D