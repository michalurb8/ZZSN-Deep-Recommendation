import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from numpy import sqrt
from numpy import argmax
from matplotlib import pyplot
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names

if __name__ == "__main__":
    # 1. Load and prepare dataset
    ratings = pd.read_csv('./Ratings.csv', 
                        dtype={'User-ID': np.int32,'ISBN': str, 'Book-Rating': np.int16 })
    users = pd.read_csv('./Users.csv', dtype={'User-ID': np.int32,
                                            'Location': str,
                                            'Age': np.float64})

    books = pd.read_csv('./Books.csv', dtype={'ISBN': str, 
                            'Book-Title':str,
                            'Book-Author':str,
                            'Year-Of-Publication': np.int16,
                            'Publisher': str,
                            'Image-URL-S': str,
                            'Image-URL-M': str,
                            'Image-URL-L': str})

    books = books.drop(columns=['Book-Title', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    ratings = ratings[ratings['Book-Rating'] != 0]
    ratings['label'] = ratings.apply(lambda x: 1 if x['Book-Rating'] > 7 else 0, axis=1)

    ratings = ratings.astype({'label': np.int8})

    data = ratings.merge(books, on='ISBN', how='left')
    data = data.merge(users, on='User-ID', how='left')
    data = data.drop(columns=['User-ID', 'ISBN', 'Book-Rating'])

    data = data.rename(columns={
        'Age': 'C1', 
        'Year-Of-Publication': 'C2',
        'Book-Author': 'C3', 
        'Publisher': 'C4', 
        'Location': 'C5'})

    data = data[['label', 'C1', 'C2', 'C3', 'C4', 'C5']]
    
    under_sampler = RandomUnderSampler(random_state=2020)
    data, y_label = under_sampler.fit_resample(data[['C1','C2','C3','C4','C5']], 
                                               data['label'])
    data['label'] = y_label
    print(f"Testing target statistics: {Counter(data['label'])}")
 
    int_features = ['C' + str(i) for i in range(1, 3)]
    str_features = ['C' + str(i) for i in range(3, 6)]
    sparse_features = ['C' + str(i) for i in range(1, 7)]
    target = ['label']

    data[str_features] = data[str_features].fillna('-1', )
    data[int_features] = data[int_features].fillna(0, )

    # 2. Transform dataset

    fixlen_feature_columns_float = [SparseFeat(feat, vocabulary_size=data[feat].nunique() * 5,
                                    embedding_dim=4, use_hash=True, dtype='float')  # since the input is numerical
                                for feat in int_features]

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique() * 5,
                                embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                              for feat in str_features]

    linear_feature_columns = fixlen_feature_columns_float + fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns_float + fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, random_state=2020)

    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
 
    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns,dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    # configure early stopping 
    es = EarlyStopping(monitor='val_binary_crossentropy')

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.1, callbacks=[es])
    model.save('./deepfm_book')
    pred_ans = model.predict(test_model_input, batch_size=256)

    fpr, tpr, thresholds = roc_curve(test[target].values, pred_ans)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='DeepFM')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    # show the plot
    pyplot.show()
    # generate labels with selected threshold
    pred_label = list(map(lambda x: 1 if x > thresholds[ix] else 0, pred_ans))
    # calculate metrics
    print("test Accuracy", round(accuracy_score(test[target].values, pred_label), 4))
    print("test Precision", round(precision_score(test[target].values, pred_label), 4))
    print("test Recall", round(recall_score(test[target].values, pred_label), 4))
    print("test F1", round(f1_score(test[target].values, pred_label), 4))
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))