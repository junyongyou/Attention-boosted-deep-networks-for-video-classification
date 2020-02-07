import cv2
import os
import numpy as np
from keras.preprocessing.image import img_to_array


class CNN_Feature_extractor():
    def __init__(self, cnn_model_name):
        self.cnn_model_name = cnn_model_name

    def extract_cnn_features_live(self, model, video_input_file_path):
        if self.cnn_model_name == 'vgg16':
            from keras.applications.vgg16 import preprocess_input
        elif self.cnn_model_name == 'vgg19':
            from keras.applications.vgg19 import preprocess_input
        elif self.cnn_model_name == 'inceptionv3':
            from keras.applications.inception_v3 import preprocess_input
        elif self.cnn_model_name == 'resnet50':
            from keras.applications.resnet50 import preprocess_input
        else:
            from keras.applications.xception import preprocess_input
        print('Extracting frames from video: ', video_input_file_path)
        vidcap = cv2.VideoCapture(video_input_file_path)
        success, image = vidcap.read()
        features = []
        success = True
        count = 0
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            if success:
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                input = preprocess_input(input)
                feature = model.predict(input).ravel()
                features.append(feature)
                count = count + 1
        unscaled_features = np.array(features)
        return unscaled_features

    def cnn_features(self, model, video_input_file_path, feature_output_file_path):
        if os.path.exists(feature_output_file_path):
            return np.load(feature_output_file_path)
        if self.cnn_model_name == 'vgg16':
            from keras.applications.vgg16 import preprocess_input
        elif self.cnn_model_name == 'vgg19':
            from keras.applications.vgg19 import preprocess_input
        elif self.cnn_model_name == 'inceptionv3':
            from keras.applications.inception_v3 import preprocess_input
        elif self.cnn_model_name == 'resnet50':
            from keras.applications.resnet50 import preprocess_input
        else:
            from keras.applications.xception import preprocess_input
        count = 0
        print('Extracting frames from video: ', video_input_file_path)
        vidcap = cv2.VideoCapture(video_input_file_path)
        success, image = vidcap.read()
        features = []
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            if success:
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                input = preprocess_input(input)
                feature = model.predict(input).ravel()
                features.append(feature)
                count = count + 1
        unscaled_features = np.array(features)
        np.save(feature_output_file_path, unscaled_features)

    def extract_cnn_features(self, data_dir_path, output_dir_path, model, data_set_name):
        input_data_dir_path = data_dir_path + '/' + data_set_name
        output_feature_data_dir_path = data_dir_path + '/' + output_dir_path

        if not os.path.exists(output_feature_data_dir_path):
            os.makedirs(output_feature_data_dir_path)

        y_samples = []
        x_samples = []

        for f in os.listdir(input_data_dir_path):
            file_path = input_data_dir_path + os.path.sep + f
            if not os.path.isfile(file_path):
                output_dir_name = f
                output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)
                for ff in os.listdir(file_path):
                    video_file_path = file_path + os.path.sep + ff
                    output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                    if not os.path.isfile(output_feature_file_path):
                        self.cnn_features(model, video_file_path, output_feature_file_path)
                    x = output_feature_file_path
                    y = f
                    y_samples.append(y)
                    x_samples.append(x)

        return x_samples, y_samples

