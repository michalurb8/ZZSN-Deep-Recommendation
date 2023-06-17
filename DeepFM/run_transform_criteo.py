import dask.dataframe as dd
from dask_ml.preprocessing import MinMaxScaler, OrdinalEncoder



if __name__ == "__main__":

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    # 1. Load data to dask dataframe with fixes for mistakes that automatic type inference makes
    def load_data(path):
        data = dd.read_csv(path, sep='	',dtype={'I11': 'float64',
        'I5': 'float64',
        'I7': 'float64',
        'I8': 'float64',
        'I9': 'float64',
        'I10': 'float64',
        'I6': 'float64',})

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
    
    # 2. Encode atributes:
    # Label Encoding for sparse features.
    # Used Ordinal Encoding - does the same thing but its better for atributes and label encoder for target labels
    # Simple Transformation for dense features
    def encode_sparse(data_train):
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
        data_train[dense_features] = tmp_dense_feat
        return data_train
    

    print("Pre encode sparse")
    data_train = encode_sparse(data_train)
    print("Pre encode dense")  
    data_train = encode_dense(data_train)
    print("Post encode")

    # 3. Save transformed dataset to CSV
    # Makes dataset a few GB smaller so its possible to train the model with less RAM
    # Dataset loading and transformation with pandas and sklearn takes a ton of time,
    # and fails on a machine with 16 GB of RAM.
    data_train.to_csv('./transformed.csv', 
                      single_file=True, 
                      header_first_partition_only=True,
                      index=False)