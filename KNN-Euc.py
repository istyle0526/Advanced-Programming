    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sn
    from sklearn.metrics import confusion_matrix

    data = pd.read_csv('./Data/wine.csv')
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    ndata = data.shape[0]
    ncolumn = data.shape[1]

    train_rate = 0.7
    ntrain = int(ndata * train_rate)
    train_index = range(ntrain)
    test_index = range(ntrain, ndata)

    train, test = data.iloc[train_index,], data.iloc[test_index,]
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]

    train_accuracy = []
    test_accuracy = []

    train_precision = []
    test_precision = []

    train_recall = []
    test_recall = []

    train_specificity= []
    test_specificity= []

    train_f1measure= []
    test_f1measure= []

    train_gmean= []
    test_gmean = []


    for k in range (1,12,2):
        KNN_R = KNeighborsClassifier(n_neighbors=k, p=2)
        KNN_R.fit(train_x, train_y)
        estimates = KNN_R.predict(train_x)
        estimates2 = KNN_R.predict(test_x)

        C = confusion_matrix(train_y, estimates)
        TN, FP, FN, TP = C.ravel()
        Accuracy = accuracy_score(train_y,estimates)
        Precision = float(TP / (TP + FP))
        Recall = float(TP / (TP + FN))
        Specificity = float(TN / (TN + FP))
        F1measure = float(2 * Precision * Recall / (Precision + Recall))
        Gmean = float(np.sqrt(Precision * Recall))

        Accuracy2 = accuracy_score(test_y, estimates2)
        C2 = confusion_matrix(test_y, estimates2)
        TN2, FP2, FN2, TP2 = C2.ravel()
        Precision2 = float(TP2 / (TP2 + FP2))
        Recall2 = float(TP2 / (TP2 + FN2))
        Specificity2 = float(TN / (TN + FP))
        F1measure2 = float(2 * Precision2 * Recall2 / (Precision2 + Recall2))
        Gmean2 = float(np.sqrt(Precision2 * Recall2))

        train_accuracy.append(Accuracy)
        test_accuracy.append(Accuracy2)

        train_precision.append(Precision)
        test_precision.append(Precision2)

        train_recall.append(Recall)
        test_recall.append(Recall2)

        train_specificity.append(Specificity)
        test_specificity.append(Specificity2)

        train_f1measure.append(F1measure)
        test_f1measure.append(F1measure2)

        train_gmean.append(Gmean)
        test_gmean.append(Gmean2)


    fig, ax = plt.subplots()
    ax.plot(range(1,12,2),train_recall, color ='black', marker='o', label = 'G-mean of train')
    ax.plot(range(1,12,2),test_recall, color ='blue', marker='x', label = 'G-mean of test')
    plt.legend()
    plt.xlabel('Depth of the tree')
    plt.ylabel('G-mean')
    ax.set(xticks=range(1,12,2),xlim=[0,13],ylim=[0,1])

    optimal_k = int(2*np.argmax(test_gmean)+1)

    KNN_O = KNeighborsClassifier(n_neighbors=optimal_k, p=2)

    KNN_O.fit(train_x,train_y)
    optimal_estimates = KNN_O.predict(train_x)
    optimal_estimates_2 = KNN_O.predict(test_x)


    C=confusion_matrix(train_y,optimal_estimates)
    TN, FP, FN, TP = C.ravel()

    Optimal_accuracy = accuracy_score(train_y, optimal_estimates)
    Precision=float(TP/(TP+FP))
    Recall=float(TP/(TP+FN))
    Specificity=float(TN/(TN+FP))
    F1measure=float(2*Precision*Recall/(Precision+Recall))
    Gmean=float(np.sqrt(Precision*Recall))


    print("optimal k is %.3f"%(optimal_k))
    print("\n"
          "\n"
          "This solution is computed using train data")
    print("Accuracy using test data is: %.3f"%(float(max(train_accuracy))))
    print(C)
    print("Precision : %.3f, Recall : %.3f, Specificity : %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision, Recall, Specificity, F1measure, Gmean))
    print("Type 1 error : %.3f, Type 2 error : %.3f"%(1-Specificity, 1-Recall))

    C_2 =confusion_matrix(test_y,optimal_estimates_2)
    TN_2, FP_2, FN_2, TP_2 = C_2.ravel()

    Optimal_accuracy_2=accuracy_score(test_y,optimal_estimates_2)
    Precision_2=float(TP_2/(TP_2+FP_2))
    Recall_2=float(TP_2/(TP_2+FN_2))
    Specificity_2=float(TN_2/(TN_2+FP_2))
    F1measure_2=float(2*Precision_2*Recall_2/(Precision_2+Recall_2))
    Gmean_2=float(np.sqrt(Precision_2*Recall_2))

    print("\n"
          "\n"
          "This solution is computed using test data")
    print(C_2)
    print("Accuracy using train data is: %.3f"%(Optimal_accuracy_2))
    print("Precision : %.3f, Recall : %.3f, Specificity: %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision_2, Recall_2, Specificity_2, F1measure_2, Gmean_2))
    print("Type 1 error : %.3f, Type 2 error : %.3f"%(1-Specificity_2, 1-Recall_2))

    df_cm = pd.DataFrame(C, ['Actual N','Actual P'],['Predicted N','Predicted P'])
    df_cm2 = pd.DataFrame(C_2, ['Actual N','Actual P'],['Predicted N','Predicted P'])

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set(title='Confusion Matrix of Train Data')

    ax2 = fig.add_subplot(212)
    ax2.set(title='Confusion Matrix of Test Data')

    sn.heatmap(df_cm, annot=True, fmt='d', ax=ax1, annot_kws={"size": 16})
    sn.heatmap(df_cm2, annot=True, fmt='d', ax=ax2, annot_kws={"size": 16})
    plt.tight_layout()
    plt.show()

for i in df_users.index:
    if df_users.loc[i]['zip']<10000:
        if zip_dict[int('0'+str(df_users.loc[i]['zip'])[:2])] <= 20000:
            df_users.loc[i]['zip'] = 0
        else:
            df_users.loc[i]['zip'] = int('0'+str(df_users.loc[i]['zip'])[:2])
    else:
        if zip_dict[int(str(df_users.loc[i]['zip'])[:3])] <= 20000:
            df_users.loc[i]['zip'] = 0
        else:
            df_users.loc[i]['zip'] = int(str(df_users.loc[i]['zip'])[:3])