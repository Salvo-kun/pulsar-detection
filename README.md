# pulsar-detection
Analysis and classification of a set of potential pulsars according to different techniques.
The dataset is split into two files: Train.txt and Test.txt, inside the folder data.
The analyzed pre-processing techniques are:
-	Z-normalization
-	Quantile transformation
- 	PCA

The analyzed models are:
-	MVG (Full, Naive Bayes, Tied, Tied Naive Bayes)
-	Prior Weighted Logistic Regression (Linear and Quadratic, through feature expansion)
-	Balanced SVM (Linear, Quadratic kernel, RBF Kernel)
-	GMM (Full, Diagonal, Tied, Tied Diagonal)
