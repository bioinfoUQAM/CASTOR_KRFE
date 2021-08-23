# CASTOR-KRFE
* CASTOR-KRFE v1.2 Help file																		  
* K-mers based feature extractor for viral genomic classification                               
* Copyright (C) 2021  Dylan Lebatteux, Amine M. Remita, Abdoulaye Banire Diallo    
* Author : Dylan Lebatteux, Amine M. Remita													  
* Contact : lebatteux.dylan@courrier.uqam.ca

### Description
CASTOR-KRFE is an alignment-free method for extracting a set of features based on k-mers to discriminate between groups of genomic sequences. The core of CASTOR-KRFE is based on feature elimination using Support Vector Machines (SVM-RFE) which is an machine learning feature selection method. CASTOR-KRFE identifies an optimal length of k to maximize classification performance and minimize the number of features. The extracted set of k-mers can be used to build a prediction model. Finally, this model can be used to predict a set of new genomic sequences. 

### Required softwares
* [python](https://www.python.org/downloads/) 
* [scikit-learn](https://scikit-learn.org/stable/install.html) 
* [numpy](https://numpy.org/install/)                        
* [biopython](https://biopython.org/wiki/Download)    

### Parameters
List of parameters requiring adjustment in the configuration_file.ini :
* k_min : Minimum length of k-mers
* k_max : Maximum length of k-mers
* T : Percentage performance threshold (T = 0.999 is recommended) .
* training_fasta : Training fasta file path
* testing_fasta : Testing fasta file path
* k_mers_path : Path file of the extracted k-mers
* model_path : Path file of the prediction model
* prediction_path : Path of the sequence prediction file
* evaluation_mode : Evaluation mode during the prediction (True/False). 

### Utilization
1) Specify the parameters of the previous section in the configuration_file.ini.
2) Run the following command :
```sh
$ python main.py configuration_file.ini
```
3) Select an option:
- 1.Extract k-mers | Required parameters: T, k_min, k_max, training_fasta and k_mers_path
- 2.Fit a model | Required parameters: training_fasta, k_mers_path and model_path
- 3.Predict a sequences | Required parameters: testing_fasta, k_mers_path, model_path, prediction_path and evaluation_mode
- 4.Exit/Quit

### Fasta file format example for n sequences: 

```sh
>id_sequence_1|target_sequence_1 
CTCAACTCAGTTCCACCAGGCTCTGTTGGATCCGAGGGTAAGGGCTCTGTATTTTCCTGC 
>id_sequence_2|target_sequence_2						
CTCAACTCAGTTCCACCAGGCTCTGTTGGATCCGAGGGTAAGGGCTCTGTATTTTCCTGC
...
...
...
>>id_sequence_n-1|target_sequence_n-1									 
CTCAACTCAGTTCCACCAGGCTCTGTTGGATCCGAGGGTAAGGGCTCTGTATTTTCCTGC 
>id_sequence_n|target_sequence_n													 
CTCAACTCAGTTCCACCAGGCTCTGTTGGATCCGAGGGTAAGGGCTCTGTATTTTCCTGC 
```
* The character "|" is used to separate the sequence ID from its target. 
* The target must be specified in the fasta file for a prediction with evaluation_mode = True.
* For more detailed examples see the data sets in the Data folder   

### Output    
* k_mers.fasta: File of the extracted k-mers list 
* model.pkl : Prediction model generated by CASTOR-KRFE                                        
* Prediction.csv : Results file of the prediction of unknown genomic sequences      

### Reference to cite CASTOR-KRFE
* [Lebatteux, D., Remita, A. M., & Diallo, A. B. (2019). Toward an alignment-free method for feature extraction and accurate classification of viral sequences. Journal of Computational Biology, 26(6), 519-535.](https://www.liebertpub.com/doi/pdfplus/10.1089/cmb.2018.0239)
                                                                                  
