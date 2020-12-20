# ML-project
Following is the Machine Learning Project whihc works on Patients ICU data depending on previous comorbidities. 
Main objective of the code is to train ML models on the dataset and predict whether or not a patient should be adimitted to ICU.
The predictions are made based on the Current Medical conditions of the patient as well as his/her previous Comorbidities like asthama, diabetes, etc. 
Different models including RandomForestClassifier, NaiveBayer, SVM, XGB regressor, etc are used along with differene ensemble techniques like bagging,
boosting and stacking.

Files Description:-
--> "True_data.csv" contains the final preprocessed dataset with clustered duplicate records and feature selection.

--> "Code_Group_27.ipynb" is the Jupyter Notebook file which contains the python code for the above project.

--> "Best_model_NB.sav" is the Naive Bayes model object saved using pickle dump. This model is already trained and need an input
                        as parameters in its predict function.  

--> "Best_model_SVM.sav" is the SVM model object saved using pickle dump. This model is already trained and need an input
                        as parameters in its predict function.

--> "Code_Group_27.py" is the Python script file for the above project. It contains all the work done for trraining and
                       visualising models as well as the dataset.

--> "visual.py" is the python file which reads the given dataset and Perform PCA,SVD, as well as t-SNE on the given Dataset. 
                   
Instructions to use:
-->Download the whole repository (all the file to avoid any errors).
-->Open and run Code_Group_27.py file to get plots and accuracy output for different models.
-->To get t-SNE plots or perform PCA and SVD, open and run visual.py file.
