import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


def tronsform_data_frame(path,path2):
    file=open(path2,'w')
    file.write('ID,idpr,target\n')
    tab=['P5DA','RIBP','8NN1','7POT','66FJ','GYSR','SOP4','RVSZ','PYUQ','LJR9','N2MW','AHXO','BSTQ','FM3X','K6QO','QBOL','JWFN','JZ9D','J9JW','GHYX','ECY3']
    data=open(path,'r')
    list=data.readlines()

    for l in range(1,len(list)):
        var = list[l]
        data = var.split(',')
        print(data)
        ch=data[0]
#        for i in range(1,8):
#           ch=ch+','+data[i]

        for j in range(8,29):
            ch1=ch+','+tab[j-8]+','+data[j][0]+'\n'
            file.write(ch1)


tronsform_data_frame('data/Train.csv','data/transformed_dataframe.csv')

path = "data/transformed_dataframe.csv"
dataframe = pd.read_csv(path)

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

path = "data/transformed_test_dataframe.csv"
dataframe_test = pd.read_csv(path)
test_ds = dataframe_to_dataset(train_dataframe)

tronsform_data_frame('data/Test.csv','data/transformed_test_dataframe.csv')

path = "data/transformed_test_dataframe.csv"
dataframe_test = pd.read_csv(path)
test_ds = dataframe_to_dataset(dataframe_test)

test_ds=test_ds.batch(32)


from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
    return encoded_feature



# Categorical features encoded as integers
target = keras.Input(shape=(1,), name="sex", dtype="int64")


# Categorical feature encoded as string
#sex = keras.Input(shape=(1,), name="sex", dtype="string")
#marital_status = keras.Input(shape=(1,), name="marital_status", dtype="string")

#branch_code = keras.Input(shape=(1,), name="branch_code", dtype="string")
#occupation_code = keras.Input(shape=(1,), name="occupation_code", dtype="string")
#occupation_category_code = keras.Input(shape=(1,), name="occupation_category_code", dtype="string")
id = keras.Input(shape=(1,), name="ID", dtype="string")
idpr=keras.Input(shape=(1,), name="idpr", dtype="string")
# Numerical features
#birth_year = keras.Input(shape=(1,), name="birth_year")


all_inputs = [
    #birth_year,
   # sex,
    #marital_status,
    #branch_code,
    #occupation_category_code,
    #occupation_code,
    id,
    idpr
]



# String categorical features
#sex_encoded = encode_string_categorical_feature(sex, "sex", train_ds)
#marital_status_encoded = encode_string_categorical_feature(marital_status, "marital_status", train_ds)
#branch_code_encoded = encode_string_categorical_feature(branch_code, "branch_code", train_ds)
#occupation_category_code_encoded = encode_string_categorical_feature(occupation_category_code, "occupation_category_code", train_ds)
#occupation_code_encoded = encode_string_categorical_feature(occupation_code, "occupation_code", train_ds)
id_encoded = encode_string_categorical_feature(id, "ID", train_ds)
idpr_encoded = encode_string_categorical_feature(idpr, "idpr", train_ds)

# Numerical features
#birth_year_encoded = encode_numerical_feature(birth_year, "birth_year", train_ds)

'''
all_features = layers.concatenate(
    [
        sex_encoded,
        marital_status_encoded,
        branch_code_encoded,
        occupation_category_code_encoded,
        occupation_code_encoded,
        id_encoded,
        idpr_encoded,
        birth_year_encoded
    ]
)
'''

all_features = layers.concatenate(
    [
        id_encoded,

        idpr_encoded,
    ]
)
x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
y = layers.Dense(128,activation="relu")(x)
y = layers.Dropout(0.5)(y)
y1 = layers.Dense(128,activation="relu")(y)
y1 = layers.Dropout(0.5)(y1)
output = layers.Dense(1, activation="sigmoid")(y1)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=3, batch_size=32,validation_data=val_ds)


file=open("data/transformed_test_dataframe.csv","r")
list= file.readlines()

list_id_x_pcode=[]

for j in range(1,len(list)):
    var = list[j].split(',')
    ch=var[0]+' X '+var[1]
    list_id_x_pcode.append(ch)
print(list_id_x_pcode)

pred = model.predict(test_ds)
pred_final=[]
for i in range(0,len(pred)):
    var=pred[i][0]
    if var>=0.5:
        var=1
    else:
        var=0
    var=str(var)
    pred_final.append(var)

print(pred)

Data = {'ID X PCODE':list_id_x_pcode,'Label':pred_final}
df = pd.DataFrame(data=Data)
df.to_csv('data_sub1.csv',index=False)