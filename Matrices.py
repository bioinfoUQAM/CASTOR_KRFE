# Function of feature and class matrix generation
def generateMatrice(data, K_mer, k):
	# Variables
	X = []
	y = []

	# Generate K-mer dictionnary
	X_dict = {}
	for i, e in enumerate(K_mer):  X_dict[e] = 0;
	# Generates X (matrix attributes)
	for d in data:
		x = []
		x_dict =  X_dict.copy()

		# Count K-mer occurences (with overlaping)
		for i in range(0, len(d[1]) - k + 1, 1):
			try: x_dict[d[1][i:i + k]] = x_dict[d[1][i:i + k]] + 1; 
			except: pass
		
		# Get only occurences from dictionnary
		for value in x_dict:
			x.append(x_dict.get(value))
		X.append(x)

	# Generates y (Matrix class) if csv file exist
	if len(data[0]) == 3: 
		for i in data: y.append(i[2])
	
	# Return matrices X and y  (matrix attributes and matrix class)
	return X, y






