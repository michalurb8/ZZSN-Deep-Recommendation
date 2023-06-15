from json import load
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from dask_ml.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder

#from deepctr.models import DeepFM
#from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

"""
dtype={'I1': 'int64',
       'I2': 'int64',
       'I3': 'int64',
       'I4': 'int64',
       'I5': 'int64',
	   'I6': 'int64',
	   'I7': 'int64',
	   'I8': 'int64',
	   'I9': 'int64',
	   'I10': 'int64',
	   'I11': 'int64',
	   'I12': 'int64',
	   'I13': 'int64'}
	   """

if __name__ == "__main__":

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    def load_data(path):
        data = dd.read_csv(path, sep='	',dtype={'I11': 'float64',
        'I5': 'float64',
        'I7': 'float64',
        'I8': 'float64',
        'I9': 'float64',
        'I10': 'float64',
        'I6': 'float64',}).head(100000,npartitions=10)
        data = dd.from_pandas(data, npartitions=10).reset_index()
        #print(data)

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )
        

        data = data.astype({'label': 'int32','I1': 'int64',
        'I2': 'int64',
        'I3': 'int64',
        'I4': 'int64',
        'I5': 'int64',
        'I6': 'int64',
        'I7': 'int64',
        'I8': 'int64',
        'I9': 'int64',
        'I10': 'int64',
        'I11': 'int64',
        'I12': 'int64',
        'I13': 'int64'})
        return data

    data_train = load_data('./train.txt')

    def encode_sparse(data_train):
        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        ##for feat in sparse_features:
            ##lbe = LabelEncoder()
            ##lbe = lbe.partial_fit(data_train[feat])
            ##print('tetett')
            #train
            ##train_sparse_feature = lbe.transform(data_train[feat])
            #print(train_sparse_feature)
            #train_sparse_feature = da.from_array(train_sparse_feature)
            #print(train_sparse_feature.compute()[99999])
            #train_sparse_feature = dd.from_dask_array(train_sparse_feature)
            #train_sparse_feature = train_sparse_feature.reset_index()
            #train_sparse_feature = train_sparse_feature.repartition(npartitions=data_train.npartitions)
            #train_sparse_feature = train_sparse_feature.reset_index()
            #train_sparse_feature.set_index(npartitions=data_train.npartitions)
            ##data_train[feat] = train_sparse_feature[0]
            #print(data_train[feat].compute())
            ##print(f'Processed: {feat}')
        oenc = OrdinalEncoder()
        train_sparse_features = data_train[sparse_features].categorize(columns=sparse_features)
        oenc = oenc.fit(train_sparse_features)
        train_sparse_features = oenc.transform(train_sparse_features)
        data_train[sparse_features] = train_sparse_features
        return data_train
        
	
    def encode_dense(data_train):
        mms = MinMaxScaler(feature_range=(0, 1))
        mms = mms.fit(data_train[dense_features])
        tmp_dense_feat = mms.transform(data_train[dense_features])
        #print(arr[11][len(arr[11]-1)])
        #tmp_dense_feat = da.from_array(arr)
        #tmp_dense_feat = dd.from_dask_array(tmp_dense_feat)
        #tmp_dense_feat = tmp_dense_feat.reset_index()
        #tmp_dense_feat = tmp_dense_feat.repartition(npartitions=data_train.npartitions)
        data_train[dense_features] = tmp_dense_feat
        
        return data_train
    

    print("Pre encode sparse")
    data_train = encode_sparse(data_train)
    print("Pre encode dense")  
    data_train = encode_dense(data_train)
    # 2.count #unique features for each sparse field,and record dense feature field name
    print("Post encode")

    data_train = data_train.round(5)

    data_train.to_csv('./transformed.csv', 
                      single_file=True, 
                      header_first_partition_only=True,
                      index=False)
    exit(0)
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=int(data_train[feat].max().compute()) + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]
    print("Post max compute")
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    #data_train.compute()
    #print(data_train.isnull())
    #train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train, test, validate = data_train.random_split([0.8, 0.1, 0.1], random_state=2020)
    train.reset_index()
    test.reset_index()
    validate.reset_index()
    train.fillna(0)
    test.fillna(0)
    validate.fillna(0)
    train_model_input = {name: train[name].compute() for name in feature_names}
    test_model_input = {name: test[name].compute() for name in feature_names}
    validate_model_input = {name: validate[name].compute() for name in feature_names}
    print("Post data split")
    #print(linear_feature_columns)
    #print(dnn_feature_columns)
    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )
    
    print(test_model_input)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_data=(validate_model_input, validate[target].values) )
    pred_ans = model.predict(test_model_input, batch_size=256)
    #print("test Accuracy", round(accuracy_score(test[target].values, pred_ans), 4))
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
