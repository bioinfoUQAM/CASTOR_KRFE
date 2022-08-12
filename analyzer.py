
##########################################################################################
### Resources                                                                          ###
### https://biopython.org/docs/1.75/api/Bio.pairwise2.html                             ###
### https://biocheminsider.com/difference-between-local-and-global-sequence-alignment/ ###
##########################################################################################

###############
### Imports ###
###############
import os
import re
import data
import kmers
import numpy
import statistics
import pandas as pd
from Bio import SeqIO
from Bio import motifs
from Bio.Seq import Seq
from Bio import Align
from Bio import pairwise2
from Bio.Seq import transcribe
from joblib import Parallel, delayed
from Bio.pairwise2 import format_alignment

#################################################################################################################################
### Function to compute perfect local pairwise sequence alignment to identify the sequences that contain the original pattern ###
#################################################################################################################################
def perfectLocalPairwiseSequenceAlignment(data, query):
     # Get the identifier
    sequence_id = data[0]
    # Get the current sequence
    sequence = data[1]
    # If the pattern is present in the sequence
    if sequence.count(query):
        # Get the position of the k-mer
        index = sequence.find(query)
        # Return the informations
        return {"id": sequence_id, "type": data[2], "pattern": query, "position": index}
    # If the pattern is not present in the sequence
    else: return {"id": sequence_id, "type": data[2], "pattern": None, "position": None}

#########################################################################################
### Function to compute local pairwise sequence alignment to identify mutated pattern ###
#########################################################################################
def localPairwiseSequenceAlignmentMutatededMotif(records, data, query, position):
    # Iterate through sequence records
    for record in records:
        # Check the corresponding sequence
        if record["id"] == data[0]: 
             # If the information are not complete 
            if record["pattern"] == None: 
                # Compute the margin
                margin = int(len(data[1]) * 10 / 100)
                # Correct margin if it is too large
                if margin > 1000 and len(data[1]) < 100000: margin = 1000
                if margin > 1000 and len(data[1]) >= 100000: margin = 2500
                # Get the position of interest
                start = position - margin
                end = position + margin
                # Correct position if out of range
                if start < 0: start = 0
                if end > len(data[1]): end = len(data[1])
                # Get the subsequence
                sequence = data[1][start:end]
                # Try to identify subsitution
                alignments = pairwise2.align.localms(sequence, query, 5, -2, -100, -0.1) 
                # Get the length of the alignement
                alignment_length = len(alignments[0].seqA[alignments[0].start:alignments[0].end])
                # Set the maximum number of substitutions that can occur 
                maximum_subsitutions_number = 3
                # Compute the threshold score
                threshold_score = (len(query) - maximum_subsitutions_number) * 5 - (maximum_subsitutions_number * 5)

                ##################################
                ### Perfect case subsititution ###
                ##################################
                if alignment_length == len(query) and alignments[0].score >= threshold_score:
                    # Get the mutated k-mer
                    pattern = alignments[0].seqA[alignments[0].start:alignments[0].end]
                    # Get the position of the mutated k-mers
                    position =  data[1].find(pattern)
                    # Return the informations
                    return {"id": record["id"], "type": data[2], "pattern": pattern, "position":position}
                
                ###########################
                ### If one substitution ###
                ###########################
                elif (len(query) - alignment_length == 1 and alignments[0].score >= threshold_score - 5):
                    # Get the index to localise where are the subsititution
                    index = query.find(alignments[0].seqB[alignments[0].start:alignments[0].end])
                    start = alignments[0].start - index
                    end = start + len(query) 
                    pattern = alignments[0].seqA[start:end] 
                    # Get the position of the mutated k-mers
                    position =  data[1].find(pattern)
                    return {"id": record["id"], "type": data[2], "pattern": pattern, "position": position}

                ############################
                ### If two substitutions ###
                ############################
                elif (len(query) - alignment_length == 2 and alignments[0].score >= threshold_score - 10):
                    
                    # Get the index to localise where are the subsititution
                    index = query.find(alignments[0].seqB[alignments[0].start:alignments[0].end])
                    start = alignments[0].start - index
                    end = start + len(query) 
                    pattern = alignments[0].seqA[start:end] 
                    # Get the position of the mutated k-mers
                    position =  data[1].find(pattern)
                    return {"id": record["id"], "type": data[2], "pattern": pattern, "position": position}
                
                ############################################################
                ### If no solution, try to identify deletion / insertion ###
                ############################################################
                else: 
                    # Try to identify deletion / insertion
                    alignments = pairwise2.align.localms(sequence, query, 1, -2, -5, -0.1)
                      
                    # Perfect deletion/insertion
                    if (alignments[0].end - alignments[0].start) == len(query) and alignments[0].score > (len(query)/2 -2):
                        position = data[1].find(alignments[0].seqA.replace("-", "")) + alignments[0].start
                        return {"id": record["id"], "type": data[2], "pattern": alignments[0].seqA[alignments[0].start:alignments[0].end], "position": position}
                    
                    # Deletion/Insertion at the begining/ending of the sequence
                    elif (alignments[0].seqA[alignments[0].start:alignments[0].end] == alignments[0].seqB[alignments[0].start:alignments[0].end]
                    and alignments[0].score  > len(alignments[0].seqA[alignments[0].start:alignments[0].end]) - 1):
                        index_subsequence = query.find(alignments[0].seqA[alignments[0].start:alignments[0].end])
                        n_start_gap = index_subsequence
                        n_end_gap = len(query) - (n_start_gap + len(alignments[0].seqA[alignments[0].start:alignments[0].end]))
                        subsequence = '-' * n_start_gap + alignments[0].seqA[alignments[0].start:alignments[0].end] + '-' * n_end_gap
                        position = data[1].find(alignments[0].seqA.replace("-", "")) + alignments[0].start
                        return {"id": record["id"], "type": data[2], "pattern": subsequence, "position": position}
                    # If no result
                    else: return {"id": record["id"], "type": data[2], "pattern": None, "position": None}
            # If the information are already complete  
            else: return record

##########################################################
### Function to indentify mutation at amino acid level ###
##########################################################
def getMutation(seqA, seqB):
    mutations = []
    alignments = pairwise2.align.globalms(seqA, seqB.replace('*',''), 2, -1, -10, -10)
    # Silent mutation
    if seqA == seqB.replace('*',''):
        mutations.append("Silent mutation")
    # If mutattion
    else:
        #print(format_alignment(*alignments[0]))
        for i, aa in enumerate(range(0, len(alignments[0].seqA))):
            # Insertion
            if alignments[0].seqA[i] != alignments[0].seqB[i] and  alignments[0].seqA[i] == "-":
                mutations.append("Insertion: " + str(i + 1) + " (" + alignments[0].seqB[i] + ")")

            # Deletion 
            elif alignments[0].seqA[i] != alignments[0].seqB[i] and  alignments[0].seqB[i] == "-":
                mutations.append("Deletion: " + str(i + 1))

            # Substitution 
            elif alignments[0].seqA[i] != alignments[0].seqB[i]:
                mutations.append("Substitution: " + alignments[0].seqA[i] + str(i + 1) +  alignments[0].seqB[i])
            else: pass
    # Return mutations
    return mutations

#####################################################################################
### Function to indentify perfect matches between initial signature and sequences ###
#####################################################################################
def identifyPerfectMatch(parameters): 
    # Display information
    print("\nIndentify perfect matches...")
    # Initialize the results list
    Results = {}
    # Get the discriminative motifs
    Kmers = kmers.loadKmers(str(parameters["k_mers_path"]))
    # Get the sequence dataset
    Data = data.loadData(str(parameters["training_fasta"]))
    # Add the reference sequence
    Data = data.loadReferenceSequence(Data, str(parameters["refence_sequence_genbank"]))
    # Iterate through the k-mers
    for kmer in Kmers:
        # Display the current motif
        print("Signature: " + kmer)
        # Get the current k-mer
        query = kmer
        # Check if there is perfect pairwise alignment of the current kmer with each sequence using parallelization 
        informations = Parallel(n_jobs = -1)(delayed(perfectLocalPairwiseSequenceAlignment)(data, query) for data in Data)
        # Save the informations of each sequence according to the current kmer
        Results[kmer] = informations
    # Return the list of dictionary
    return Results

#########################################################################################
### Function to indentify variations with sequences not associated with perfect match ###
#########################################################################################
def identifyVariations(Results, parameters): 
    # Display information
    print("\nIndentify variations...")
    # Get the sequence dataset
    Data = data.loadData(str(parameters["training_fasta"]))
    # Add the reference sequence
    Data = data.loadReferenceSequence(Data, str(parameters["refence_sequence_genbank"]))

    # Iterate through the actual sequences records
    for key, records in Results.items():
        print("Signature: " + key)
        # Get the current k-mer
        query = key
        # Compute the average position of the k-mers to prune the search for mutated k-mers 
        positions = []
        for record in records:
            if record["position"] != None: positions.append(record["position"])
        average_position = int(statistics.mean(positions))
        # Compute the pairwise alignment of the amplified motif with each sequence to identify mutated motifs using parallelization 
        informations = Parallel(n_jobs = -1)(delayed(localPairwiseSequenceAlignmentMutatededMotif)(records, data, query, average_position) for data in Data)
        # Save the updated informations of each sequence according to the current kmer
        Results[key] = informations
    # Dictionary of mutated motif (s) and associated postion(s)
    return Results 

  
##########################################################
### Function to indentify mutation at amino acid level ###
##########################################################
def indentifyMutationInformation(record, reference_sequence_informations):
    # If same pattern as reference sequence
    if record["pattern"] == reference_sequence_informations["pattern"]:
        # Save the information associated to the reference pattern
        return [record["id"], reference_sequence_informations["organisme"], record["type"], reference_sequence_informations["pattern"], None, reference_sequence_informations["gene"], None, record["position"]]

    # If different pattern as reference sequence        
    else: 
        # If a pattern variation has been identified
        if record["pattern"] != None:
            # Get the position of the initial pattern 
                start = reference_sequence_informations["position_gene_sequence"]
                end = reference_sequence_informations["position_gene_sequence"] + len(record["pattern"])

                # Generate the mutated sequence
                mutated_squence = ""
                mutated_squence = mutated_squence + reference_sequence_informations["nucleotide_sequence"][0:start]
                mutated_squence = mutated_squence + record["pattern"].replace("-", "")
                mutated_squence = mutated_squence + reference_sequence_informations["nucleotide_sequence"][end:]
                rna_sequence = transcribe(mutated_squence)
                aa_sequence = str(rna_sequence.translate())

                # Get the mutation
                mutations = getMutation(reference_sequence_informations["amino_acid_sequence"], aa_sequence)
                if len(mutations) <= 5:
                    return[record["id"], reference_sequence_informations["organisme"],record["type"],reference_sequence_informations["pattern"],
                    record["pattern"], reference_sequence_informations["gene"],mutations,record["position"]]
                else: 
                    return[record["id"], reference_sequence_informations["organisme"], record["type"], reference_sequence_informations["pattern"], "Not identified", "Not identified", "Not identified", "Not identified"]
                                    
        # If unidentified pattern
        else: 
            return[record["id"], reference_sequence_informations["organisme"], record["type"], reference_sequence_informations["pattern"], "Not identified", "Not identified", "Not identified", "Not identified"]


##############################################################
### Function to indentify infornation related to variation ###
##############################################################
def extractRelatedInformation (Results, parameters):
    # Display information
    print("\nExtract information related to signatures/variations...")
    # Get the reference fil path
    gb_reference_file = str(parameters["refence_sequence_genbank"])
     # Check if genbank file exists
    file_exists = os.path.exists(gb_reference_file)
    # If there is genbank file
    if file_exists == True:
        # Open the reference sequence genbank file
        for gb_record in SeqIO.parse(open(gb_reference_file, "r"), "genbank"):
            # Iterate through the sequences records for each pattern
            for key, records in Results.items():
                print("Signature: " + key)
                # Variable to check if cds was found
                isCDS = False
                # Iterate through the sequences records
                for record in records:
                    # Find the reference sequence to get the informations 
                    if record["id"] == gb_record.annotations["accessions"][0]: 
                        # Initialize the dict of reference sequence abd save initial informations
                        reference_sequence_informations = {}
                        reference_sequence_informations["id"] = record["id"]
                        reference_sequence_informations["pattern"] = record["pattern"]
                        reference_sequence_informations["position"] = record["position"]
                        reference_sequence_informations["mutated_pattern"] = None
                        reference_sequence_informations["mutation"] = None             
                        # Iterate through the features of the reference sequence
                        for feature in gb_record.features:
                            # Get the organisme informations
                            if feature.type == "source": 
                                try: reference_sequence_informations["organisme"] = feature.qualifiers["organism"][0]
                                except: reference_sequence_informations["organisme"] = None
                            # If the feature is CDS
                            if feature.type == "CDS": 
                                # Get the nucleotide sequence associated to to the CDS
                                cds = feature.location.extract(gb_record.seq)
                                # If the nucleotide sequence contains the pattern
                                if cds.find(record["pattern"]) != -1:
                                    # Set is CDS to true
                                    isCDS = True
                                    # Initialize list of data to save
                                    data = []
                                    # Get the  informations
                                    try: reference_sequence_informations["gene"] = feature.qualifiers["gene"][0]
                                    except: reference_sequence_informations["gene"] = None
                                    try: reference_sequence_informations["nucleotide_sequence"] = feature.location.extract(gb_record.seq)
                                    except: reference_sequence_informations["nucleotide_sequence"] = None
                                    try: reference_sequence_informations["position_gene_sequence"] = reference_sequence_informations["nucleotide_sequence"].find(record["pattern"])
                                    except: reference_sequence_informations["position_gene_sequence"] = None
                                    try: reference_sequence_informations["amino_acid_sequence"] = feature.qualifiers["translation"][0]
                                    except: reference_sequence_informations["amino_acid_sequence"] = None
                                    # Iterate through the sequences records to complte the infornations according to the reference informations
                                    data = Parallel(n_jobs = -1)(delayed(indentifyMutationInformation)(record, reference_sequence_informations) for record in records)
                                    # Compute the frequency
                                    for d1 in data: 
                                        organisme_type = d1[2]
                                        pattern = d1[4]
                                        n_total = 0 
                                        n_pattern = 0
                                        for d2 in data:
                                            if organisme_type == d2[2]: 
                                                n_total = n_total + 1
                                                if pattern == d2[4]: 
                                                    n_pattern = n_pattern + 1
                                        frequency = int(n_pattern / n_total * 100)
                                        d1.append(frequency)

                                    # Convert data to dataframe      
                                    df = pd.DataFrame(data, columns = ["SEQUENCE ID", "ORGANISM", "TYPE", "INITIAL PATTERN", "MUTATED PATTERN", "GENE", "MUTATIONS", "LOCATION", "FREQUENCY"])
                                    # Save the dataframe to excel file
                                    df.to_excel("output/" + key + "_gene_" + reference_sequence_informations["gene"] + ".xlsx")
                        
                        # Check for other features
                        if isCDS == False:
                            data = []
                            # Iterate through the features of the reference sequence
                            for feature in gb_record.features:
                                seq = feature.location.extract(gb_record.seq)
                                # If the nucleotide sequence contains the pattern
                                if seq.find(record["pattern"]) != -1: 
                                    # Save informations
                                    if feature.type == "rep_origin" or feature.type == "gene" or feature.type == "misc_feature" or feature.type == "repeat_region":  
                                        if feature.type == "gene":
                                            try: reference_sequence_informations["gene"] = feature.qualifiers["gene"][0]
                                            except: reference_sequence_informations["gene"] = "None"
                                        elif feature.type == "repeat_region":
                                            try: reference_sequence_informations["gene"] = "Repeat region"
                                            except: reference_sequence_informations["gene"] = "None"
                                        else:  
                                            try: reference_sequence_informations["gene"] = feature.qualifiers["note"][0]
                                            except: reference_sequence_informations["gene"] = "None"

                            # If no feature identified
                            if "gene" not in reference_sequence_informations: reference_sequence_informations["gene"] = "None"
            
                            # Iterate through the sequences records to complete the infornations according to the reference informations
                            for record in records:
                                # If same pattern as reference sequence 
                                if record["pattern"] == reference_sequence_informations["pattern"]:
                                    data.append([record["id"], reference_sequence_informations["organisme"], record["type"], reference_sequence_informations["pattern"], None, reference_sequence_informations["gene"], None, record["position"]])
                                # If different pattern as reference sequence        
                                else: 
                                    if record["pattern"] != None:
                                        data.append([record["id"], reference_sequence_informations["organisme"], record["type"],reference_sequence_informations["pattern"],record["pattern"], reference_sequence_informations["gene"], "No CDS",record["position"]])
                                        
                                    # If unidentified pattern
                                    else: 
                                        data.append([record["id"], reference_sequence_informations["organisme"], record["type"], reference_sequence_informations["pattern"], "Not identified", "Not identified", "Not identified", "Not identified"])
                                    
                            # Compute the frequency
                            for d1 in data: 
                                organisme_type = d1[2]
                                pattern = d1[4]
                                n_total = 0 
                                n_pattern = 0

                                for d2 in data:
                                    if organisme_type == d2[2]: 
                                        n_total = n_total + 1
                                        if pattern == d2[4]: 
                                            n_pattern = n_pattern + 1
                                frequency = int(n_pattern / n_total * 100)
                                d1.append(frequency) 

                            # Convert data to dataframe      
                            df = pd.DataFrame(data, columns = ["SEQUENCE ID", "ORGANISM", "TYPE", "INITIAL PATTERN", "MUTATED PATTERN", "GENE", "MUTATIONS", "LOCATION", "FREQUENCY"])
                            # Save the dataframe to excel file
                            df.to_excel("output/" + key + "_gene_" + reference_sequence_informations["gene"] + ".xlsx") 

    # If no genbank reference sequence
    else:
        # Iterate through the sequences records for each pattern
        for key, records in Results.items():
            data = []
            # Iterate through the sequences records
            for record in records: 
                data.append([record["id"], None, record["type"], key, record["pattern"], None, None, record["position"]])

            # Compute the frequency
            for d1 in data: 
                organisme_type = d1[2]
                pattern = d1[4]
                n_total = 0 
                n_pattern = 0

                for d2 in data:
                    if organisme_type == d2[2]: 
                        n_total = n_total + 1
                        if pattern == d2[4]: 
                            n_pattern = n_pattern + 1
                frequency = int(n_pattern / n_total * 100)
                d1.append(frequency)

            # Convert data to dataframe      
            df = pd.DataFrame(data, columns = ["SEQUENCE ID", "ORGANISM", "TYPE", "INITIAL PATTERN", "MUTATED PATTERN", "GENE", "MUTATIONS", "LOCATION", "FREQUENCY"])
            # Save the dataframe to excel file
            df.to_excel("output/" + key + ".xlsx")           