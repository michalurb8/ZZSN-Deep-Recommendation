import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, roc_curve, recall_score, precision_score
from sklearn.model_selection import train_test_split

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from tensorflow.python.keras.callbacks import EarlyStopping
from numpy import sqrt
from numpy import argmax
from matplotlib import pyplot

if __name__ == "__main__":
    data = pd.read_csv('./train_transformed.csv',# index_col='index', 
        dtype={'I1' : 'float16',
            'I2' : 'float16',
            'I3' : 'float16',
            'I4' : 'float16',
            'I5' : 'float16',
            'I6' : 'float16',
            'I7' : 'float16',
            'I8' : 'float16',
            'I9' : 'float16',
            'I10' : 'float16',
            'I11' : 'float16',
            'I12' : 'float16',
            'I13' : 'float16',
            'C1' : 'int32',
            'C2' : 'int32',
            'C3' : 'int32',
            'C4' : 'int32',
            'C5' : 'int32',
            'C6' : 'int32',
            'C7' : 'int32',
            'C8' : 'int32',
            'C9' : 'int32',
            'C10' : 'int32',
            'C11' : 'int32',
            'C12' : 'int32',
            'C13' : 'int32',
            'C14' : 'int32',
            'C15' : 'int32',
            'C16' : 'int32',
            'C17' : 'int32',
            'C18' : 'int32',
            'C19' : 'int32',
            'C20' : 'int32',
            'C21' : 'int32',
            'C22' : 'int32',
            'C23' : 'int32',
            'C24' : 'int32',
            'C25' : 'int32',
            'C26' : 'int32'
       })
    #exit(0)
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']
    print(data.info())
    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    es = EarlyStopping(monitor='val_binary_crossentropy')
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, callbacks=[es])
    pred_ans = model.predict(test_model_input, batch_size=256)

    
    fpr, tpr, thresholds = roc_curve(test[target].values, pred_ans)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    # show the plot
    pyplot.show()
    pred_label = list(map(lambda x: 1 if x > thresholds[ix] else 0, pred_ans))
    print("test Accuracy", round(accuracy_score(test[target].values, pred_label), 4))
    print("test Precision", round(precision_score(test[target].values, pred_label), 4))
    print("test Recall", round(recall_score(test[target].values, pred_label,), 4))
    print("test F1", round(f1_score(test[target].values, pred_label), 4))

    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))