# Import
import configparser

# Fonction to get the parameters from the configuration file
def getParameters(configuration_file):
	# Initialize the parser
	configurationParser = configparser.ConfigParser()
	# Read the configuration file
	configurationParser.read(configuration_file)
	# Get the parameters
	parameters = dict()
	parameters["T"] = configurationParser.get("parameters", "T")
	parameters["k_min"] = configurationParser.get("parameters", "k_min")
	parameters["k_max"] = configurationParser.get("parameters", "k_max")
	parameters["model_path"] = configurationParser.get("parameters", "model_path")
	parameters["k_mers_path"] = configurationParser.get("parameters", "k_mers_path")
	parameters["testing_fasta"] = configurationParser.get("parameters", "testing_fasta")
	parameters["training_fasta"] = configurationParser.get("parameters", "training_fasta")
	parameters["refence_sequence_genbank"] = configurationParser.get("parameters", "reference_sequence")
	parameters["prediction_path"] = configurationParser.get("parameters", "prediction_path")
	parameters["evaluation_mode"] = configurationParser.get("parameters", "evaluation_mode")
	# Return the parameter dictionary
	return parameters
