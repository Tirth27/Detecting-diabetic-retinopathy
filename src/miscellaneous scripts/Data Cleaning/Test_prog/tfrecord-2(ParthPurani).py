import tensorflow as tf
import os
import matplotlib.image as mpimg

class GenerateTFRecord:
    def __init__(self, labels):
        self.labels = labels

    def convert_image_folder(self, img_folder, tfrecord_file_name):
        # Get all file names of images present in folder
        img_paths = os.listdir(img_folder)
        img_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in img_paths]

        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path in img_paths:
                example = self._convert_image(img_path)
                writer.write(example.SerializeToString())

    def _convert_image(self, img_path):
        label = self._get_label_with_filename(img_path)
        image_data = mpimg.imread(img_path)
        # Convert image to string data
        image_str = image_data.tostring()
        # Store shape of image for reconstruction purposes
        img_shape = image_data.shape
        # Get filename
        filename = os.path.basename(img_path)
        
        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[2]])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_str])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
        }))
        return example

    def _get_label_with_filename(self, filename):
        basename = os.path.basename(filename).split('.')[0]
        basename = basename.split('_')[0]
        return self.labels[basename]

if __name__ == '__main__':
    labels = {'cat': 0, 'dog': 1}
    t = GenerateTFRecord(labels)
    t.convert_image_folder('Images', 'images.tfrecord')