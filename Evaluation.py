# Imports
import joblib
import Matrices
import Preprocessing
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

# Function of model evaluation
def cross_validation(training_data, extracted_k_mers, identified_k_length, data):
	# Generate matrices
	X, y = Matrices.generateMatrice(training_data, extracted_k_mers, identified_k_length)
	X = Preprocessing.minMaxScaling(X)

	# Realize evaluation with CV + Classifier + Metrics
	classifier = svm.SVC(kernel = 'linear', C = 1)
	stratifiedKFold = StratifiedKFold(n_splits = 5, shuffle = False, random_state = None)
	y_pred = cross_val_predict(classifier, X, y, cv = stratifiedKFold, n_jobs = 4)
	
	# Print results of model evaluation
	classificationReport = classification_report(y, y_pred, digits = 3)
	confusionMatrix = confusion_matrix(y, y_pred)
	print("\nClassification report of model evaluation\n", classificationReport)
	print("Confusion matrix \n", confusionMatrix)

	# Save Matrice
	f = open("Output/Matrice.csv", "w")
	f.write("Id,")
	for i in extracted_k_mers: f.write(str(i) + ","); 
	f.write("Class\n")

	for i, x in enumerate(X):
		f.write(str(data[i][0]) + ",")
		for j in x: f.write(str(j) + ",")
		f.write(str(y[i]) + "\n")
	f.close()

	# Save model
	classifier.fit(X, y)
	joblib.dump(classifier, 'Output/model.pkl') 

	# Save results of model evaluation
	f = open("Output/Model_Evaluation.txt", "w")
	f.write("Classification report of model evaluation\n" +  classificationReport);
	f.write("\nConfusion matrix \n" + str(confusionMatrix));
	f.close()



# Function of prediction without evaluation
def prediction(training_data, testing_data, extracted_k_mers, identified_k_length):
	# Generate matrices
	X_test, y_test = Matrices.generateMatrice(testing_data, extracted_k_mers, identified_k_length)
	X_test = Preprocessing.minMaxScaling(X_test)
	
	# Load model
	classifier = joblib.load('Output/model.pkl')

	# Realize prediction
	y_pred = classifier.predict(X_test)

	# Save prediction 
	f = open("Output/Prediction.csv", "w")
	f.write("id,y_pred\n");
	for i, y in enumerate(y_pred): f.write(testing_data[i][0] + "," + y + "\n");
	f.close()

# Function of prediction with evaluation
def predictionEvaluation(training_data, testing_data, extracted_k_mers, identified_k_length):
	# Generate matrices
	X_test, y_test = Matrices.generateMatrice(testing_data, extracted_k_mers, identified_k_length)
	X_test = Preprocessing.minMaxScaling(X_test)
	
	# Load model
	classifier = joblib.load('Output/model.pkl')

	# Realize prediction
	y_pred = classifier.predict(X_test)

	# Print results
	classificationReport = classification_report(y_test, y_pred, digits = 3)
	confusionMatrix = confusion_matrix(y_test, y_pred)
	print("\nClassification report of prediction evaluation\n", classificationReport)
	print("Confusion matrix \n", confusionMatrix)

	# Save prediction 
	f = open("Output/Prediction_Evaluation.csv", "w")
	f.write("id,y_pred,y_true\n");
	for i, y in enumerate(y_pred): f.write(testing_data[i][0] + "," + y + "," + y_test[i] + "\n");
	f.close()

	# Save results of prediction evaluation
	f = open("Output/Prediction_Evaluation.txt", "w")
	f.write("Classification report of prediction evaluation\n" +  classificationReport);
	f.write("\nConfusion matrix \n" + str(confusionMatrix));
	f.close()



