import tensorflow as tf

data_arr = [
    {
        'int_data': 108,
        'float_data': 2.45,
        'str_data': 'String 100',
        'float_list_data': [256.78, 13.9]
    },
    {
        'int_data': 37,
        'float_data': 84.3,
        'str_data': 'String 200',
        'float_list_data': [1.34, 843.9, 65.22]
    }
]

def get_example_object(data_record):
    # Convert individual data into a list of int64 or float or bytes
    int_list1 = tf.train.Int64List(value = [data_record['int_data']])
    float_list1 = tf.train.FloatList(value = [data_record['float_data']])
    # Convert string data into list of bytes
    str_list1 = tf.train.BytesList(value = [data_record['str_data'].encode('utf-8')])
    float_list2 = tf.train.FloatList(value = data_record['float_list_data'])

    # Create a dictionary with above lists individually wrapped in Feature
    feature_key_value_pair = {
        'int_list1': tf.train.Feature(int64_list = int_list1),
        'float_list1': tf.train.Feature(float_list = float_list1),
        'str_list1': tf.train.Feature(bytes_list = str_list1),
        'float_list2': tf.train.Feature(float_list = float_list2)
    }

    # Create Features object with above feature dictionary
    features = tf.train.Features(feature = feature_key_value_pair)

    # Create Example object with features
    example = tf.train.Example(features = features)
    return example

with tf.python_io.TFRecordWriter('example.tfrecord') as tfwriter:
    # Iterate through all records
    for data_record in data_arr:
        example = get_example_object(data_record)

        # Append each example into tfrecord
        tfwriter.write(example.SerializeToString())