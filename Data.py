# Imports
import os
import sys
import csv
from Bio import SeqIO

# Function checking the existence and accessibility of training files 
def checkTrainFile(training_fasta, training_csv):
	# Check fasta file
	if os.path.isfile(training_fasta) and os.access(training_fasta, os.R_OK): 
		print(training_fasta, "file exists and is readable")
		# Check csv file
		if os.path.isfile(training_csv) and os.access(training_csv, os.R_OK): 
			print(training_csv, "file exists and is readable")
			# Return True if all is correct
			return True
		# Exit and display a message in case of error 
		else: sys.exit("Training csv file is missing or not readable")
	else: sys.exit("Training fasta file is missing or not readable")

# Function checking the existence and accessibility of testing files 
def checkTestFile(testing_fasta, testing_csv):
	# Check fasta file
	if os.path.isfile(testing_fasta) and os.access(testing_fasta, os.R_OK): 
		print(testing_fasta, "file exists and is readable")
		# Check csv file
		if os.path.isfile(testing_csv) and os.access(testing_csv, os.R_OK): 
			# Return True if all is correct (prediction with evaluation)
			print(testing_csv, "file exists and is readable")
			return True
		else: 
			# Return True if only fasta file is correct (prediction without evaluation)
			print("Testing csv file is missing or not readable")
			return True
	else: sys.exit("Testing fasta file is missing or not readable")

# Function generating the data table 
def generateTrainData(fasta_file, csv_file):
	# Variable data 
	data = []

	# Open the class file
	with open(csv_file) as f: reader = dict(csv.reader(f))

	# Open the sequences file
	for record in SeqIO.parse(fasta_file, "fasta"):
		# Generate table [Id, Sequences, Class]
		if record.id in reader: data.append([record.id, record.seq.upper(), reader[record.id]])

	# Return data
	return data

# Function generating the data table 
def generateTestData(fasta_file, csv_file):
	# Variable data 
	data = []

	# Call classical function
	if csv_file: data = generateTrainData(fasta_file, csv_file)
	else: 
		# Open the sequences file and generate table [Id, Sequences]
		for record in SeqIO.parse(fasta_file, "fasta"): data.append([record.id, record.seq.upper()])
			
	# Return data
	return data

