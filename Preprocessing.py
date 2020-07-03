# Import
from sklearn.preprocessing import MinMaxScaler

# Function of MinMaxScaler (0, 1)
def minMaxScaling(X):
	minMaxScaler = MinMaxScaler(feature_range = (0, 1), copy = False)
	X = minMaxScaler.fit_transform(X)
	return X
