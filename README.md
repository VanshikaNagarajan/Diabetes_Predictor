# THIS IS A MACHINE LEARNING PROJECT

# DIABETES PREDICTION

page link :  "https://github.com/VanshikaNagarajan/Diabetes_Predictor"

code link : ""

website link : ""


THIS PROJECT IS TO PREDICT IF A PERSON HAS DIABETES OR NOT BASED ON THEIR MEDICAL DATA.

BASICALLY AN ML PROJECT CONSISTS OF A MODEL THAT HAS TO BE TRAINED SO THAT IT GIVES ACCURATE PREDICTIONS. 

A GOOD MODEL IS CHOSEN BY ITS ACCURACY MEASURES, SO THAT THE USER GETS AN ACCURATE RESULT.

THE LIBRARIES USED IN THIS PROJECT ARE:

1. pandas
2. train_test_split
3. RandomForestClassifier
4. mean_squared_error
5. r2_score
6. accuracy_score
7. Scikit_learn
8. classification_report

THE MODEL USED IS RandomForest

THIS PROBLEM IS TO PREDICT THE DIABETES
AND THE SOLUTION, TO THE PROBLEM IS TO USE RANDOMFOREST MODEL TO SOLVE THIS. 

A RandomForest IS A POPULAR MACHINE LEARNING ALGORITHM USED FOR CLASSIFICATION AND REGRESSION TASKS DUE TO ITS HIGH ACCURACY AND SCALABILITY. 

IT ALSO GIVES BEST SOLUTION TO OUTLIERS IN THE DATA. 

1. TO LOAD THE DATA IN csv, USING pandas

        diabetes_data = pd.read_csv('diabetes_prediction_dataset.csv')

2. TO REMOVE THE DUPLICATE ROWS, IF THERE IS ANY

        diabetes_data.drop_duplicates(inplace=True)


3. TO CONVERT THE NUMERICAL VALUE TO BINARY FORM (0 AND 1) TO EASE FOR MACHINE TO UNDERSTAND THE DATA. 

        diabetes_data = pd.get_dummies(diabetes_data, columns=['gender'], drop_first=True)
        diabetes_data = pd.get_dummies(diabetes_data, columns=['smoking_history'], drop_first=True)

4. TO SPLIT THE DATA INTO TRAIN AND TEST

        from sklearn.model_selection import train_test_split
        X = diabetes_data.drop('diabetes', axis=1)
        Y = diabetes_data['diabetes']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        

5. TO SHOW THE RATION IN WHICH IT HAS BE SPLITTED 

        print("X_train shape:", X_train.shape)
        print("Y_train shape:", Y_train.shape)
        print("X_test shape:", X_test.shape)
        print("Y_test shape:", Y_test.shape)

6. TO SELECT THE MODEL BASED ON ITS ACCURACY 

i. LINEAR REGRESSION 

 to first fit the dataset in train into the model 

    model_linear = LinearRegression()
    model_linear.fit(X_train, Y_train)

 to formulate and calculate its accuracy 


        Y_pred = model.predict(X_test)
        Y_pred_binary = [0 if pred < 0.5 else 1 for pred in Y_pred]
        mse = mean_squared_error(Y_test, Y_pred)
        accuracy = accuracy_score(Y_test, Y_pred_binary)
        r2 = r2_score(Y_test, Y_pred)
        return mse, r2, accuracy
    mse_linear, r2_linear, accuracy_linear = evaluate_model(model_linear, X_test, Y_test)

ii. RandomForestClassifier

to first fit the dataset in train into the model 

    model_rf = RandomForestClassifier()
    Y_train = Y_train.values.reshape(-1, 1)
    model_rf.fit(X_train, Y_train)
    Y_test_reshaped = Y_test.values.reshape(-1, 1)  

to formulate and calculate its accuracy 

        Y_pred = model.predict(X_test)
        Y_pred_binary = [0 if pred < 0.5 else 1 for pred in Y_pred]
        accuracy = accuracy_score(Y_test, Y_pred_binary)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        return mse, r2, accuracy
    mse_rf, r2_rf, accuracy_rf = evaluate_model(model_rf, X_test, Y_test)


iii. KNeighbhorsRegressor

to first fit the dataset in train into the model 


    from sklearn.neighbors import KNeighborsRegressor
    model_knn = KNeighborsRegressor()
    model_knn.fit(X_train, Y_train)


to formulate and calculate its accuracy 

        Y_pred = model.predict(X_test)
        Y_pred_binary = [0 if pred < 0.5 else 1 for pred in Y_pred]
        accuracy = accuracy_score(Y_test, Y_pred_binary)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        return mse, r2, accuracy
    mse_knn, r2_knn, accuracy_knn = evaluate_model(model_knn, X_test, Y_test)


iv. Support Vector Machine 

to first fit the dataset in train into the model 

    from sklearn.svm import SVR
    model_svm = SVR()
    model_svm.fit(X_train, Y_train)


to formulate and calculate its accuracy

        Y_pred = model.predict(X_test)
        Y_pred_binary = [0 if pred < 0.5 else 1 for pred in Y_pred]
        mse = mean_squared_error(Y_test, Y_pred)
        accuracy = accuracy_score(Y_test, Y_pred_binary)
        r2 = r2_score(Y_test, Y_pred)
        return mse, r2, accuracy

    mse_svm, r2_svm, accuracy_svm = evaluate_model(model_svm, X_test, Y_test)

v. Naive Bayes Classifier

to first fit the dataset in train into the model 

    from sklearn.naive_bayes import GaussianNB
    model_nb = GaussianNB()
    model_nb.fit(X_train, Y_train)

to formulate and calculate its accuracy


    Y_pred = model_nb.predict(X_test)
    Y_pred_binary = [0 if pred < 0.5 else 1 for pred in Y_pred]
    accuracy = accuracy_score(Y_test, Y_pred_binary)

vi. DECISION TREE

to first fit the dataset in train into the model 


    from sklearn.tree import DecisionTreeRegressor
    model_dt = DecisionTreeRegressor()
    model_dt.fit(X_train, Y_train)

to formulate and calculate its accuracy

        Y_pred = model.predict(X_test)
        Y_pred_binary = [0 if pred < 0.5 else 1 for pred in Y_pred]
        accuracy = accuracy_score(Y_test, Y_pred_binary)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        return mse, r2, accuracy
    mse_dt, r2_dt, accuracy_dt = evaluate_model(model_dt, X_test, Y_test)



SO BY THIS, WE CALCULATE THE ACCURACY WHICH MODEL IS BEST FOR THE USER INTERFACE.


WE'VE TAKEN RandomForestClassifier into consideration as it gives the best accuracy,

then, WE DO THE USER ITNERFACE, WHEREIN THE USER PROVIDES THE USER DATA AND THE DIABETES PREDICTOR SYSTEM PREDICTS IF HE OR SHE HAS DIABETES.