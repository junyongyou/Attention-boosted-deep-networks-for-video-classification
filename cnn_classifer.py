from keras.layers import Input, Dense, Flatten, Activation, Dropout, Bidirectional, Permute, multiply
from keras.layers.recurrent import LSTM
from keras.callbacks import CSVLogger
from keras.models import Sequential, Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import time
import numpy as np
from video_classification.cnn_feature_extractor import CNN_Feature_extractor


# BATCH_SIZE = 64
NUM_EPOCHS = 80
VERBOSE = 1
HIDDEN_UNITS = 256
# MAX_ALLOWED_FRAMES = 20
# EMBEDDING_SIZE = 100


def attention_3d_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def generate_batch(x_samples, y_samples, batch_size):
    num_batches = len(x_samples) // batch_size

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            yield np.array(x_samples[start:end]), y_samples[start:end]


class BidirectionalLSTMVideoClassifier(object):
    def __init__(self, cnn_model_name):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.cnn_model = None
        self.expected_frames = None
        self.include_top = True
        self.config = None
        self.cnn_model_name = cnn_model_name.lower()

    def create_model_no_attention(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=False),
                                input_shape=(self.expected_frames, self.num_input_tokens)))
        # model.add(Bidirectional(LSTM(10)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def create_attention_model_before(self):
        inputs = Input(shape=(self.expected_frames, self.num_input_tokens,))
        attention_inputs = attention_3d_block(inputs, self.expected_frames)
        lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=False))(attention_inputs)
        x = Dense(512, activation='relu')(lstm_out)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(input=[inputs], output=x)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # print(model.summary())
        return model

    def create_attention_model_after(self):
        inputs = Input(shape=(self.expected_frames, self.num_input_tokens,))
        lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
        attention_mul = attention_3d_block(lstm_out, self.expected_frames)
        attention_mul = Flatten()(attention_mul)
        x = Dense(512, activation='relu')(attention_mul)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(input=[inputs], output=x)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # print(model.summary())
        return model

    def load_model(self, config_file_path, weight_file_path, attention=0):
        if os.path.exists(config_file_path):
            print('loading configuration from ', config_file_path)
        else:
            raise ValueError('cannot locate config file {}'.format(config_file_path))

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.include_top = config['include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])
        self.config = config

        if attention == 1:
            self.model = self.create_attention_model_after()
        elif attention == 2:
            self.model = self.create_attention_model_before()
        else:
            self.model = self.create_model_no_attention()
        if os.path.exists(weight_file_path):
            print('loading network weights from ', weight_file_path)
        else:
            raise ValueError('cannot local weight file {}'.format(weight_file_path))

        self.model.load_weights(weight_file_path)

        if self.cnn_model_name == 'vgg16':
            self.cnn_model = VGG16(include_top=self.include_top, weights='imagenet')
        elif self.cnn_model_name == 'vgg19':
            self.cnn_model = VGG19(include_top=self.include_top, weights='imagenet')
        elif self.cnn_model_name == 'inceptionv3':
            self.cnn_model = InceptionV3(include_top=self.include_top, weights='imagenet')
        elif self.cnn_model_name == 'resnet50':
            self.cnn_model = ResNet50(include_top=self.include_top, weights='imagenet')
        else:
            self.cnn_model = Xception(include_top=self.include_top, weights='imagenet')

        self.cnn_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, video_file_path):
        feature_extractor = CNN_Feature_extractor(self.cnn_model_name)
        x = feature_extractor.extract_cnn_features_live(self.cnn_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label

    def fit(self, data_dir_path, model_dir_path, include_top=True, data_set_name='UCF-101', test_size=0.3,
            attention=0, do_feature_extraction=False):

        self.include_top = include_top
        batch_size = 128
        if self.cnn_model_name == 'vgg16':
            print('vgg16')
            self.cnn_model = VGG16(include_top=self.include_top, weights='imagenet')
        elif self.cnn_model_name == 'vgg19':
            print('vgg19')
            self.cnn_model = VGG19(include_top=self.include_top, weights='imagenet')
        elif self.cnn_model_name == 'inceptionv3':
            print('inceptionv3')
            self.cnn_model = InceptionV3(include_top=self.include_top, weights='imagenet')
        elif self.cnn_model_name == 'resnet50':
            print('resnet50')
            self.cnn_model = ResNet50(include_top=self.include_top, weights='imagenet')
            batch_size = 8
        else:
            print('xception')
            self.cnn_model = Xception(include_top=self.include_top, weights='imagenet')
            batch_size = 16

        self.cnn_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        if not include_top:
            feature_dir_name = data_set_name + '-' + self.cnn_model_name + '-HiDimFeatures'
        else:
            feature_dir_name = data_set_name + '-' + self.cnn_model_name + '-Features'
        max_frames = 0
        self.labels = dict()
        feature_extractor = CNN_Feature_extractor(self.cnn_model_name)
        x_samples, y_samples = feature_extractor.scan_and_extract_cnn_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.cnn_model,
                                                               data_set_name=data_set_name)
        if do_feature_extraction:
            return
        print(x_samples[0].shape)
        self.num_input_tokens = x_samples[0].shape[1]
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            frames_list.append(frames)
            max_frames = max(frames, max_frames)
        self.expected_frames = int(np.mean(frames_list))
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)

        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['include_top'] = self.include_top

        self.config = config

        t1 = time.time()
        if attention == 1:
            model = self.create_attention_model_after()
            csv_logger = CSVLogger('{}_attention_after.log'.format(self.cnn_model_name), append=True, separator=';')
        elif attention == 2:
            model = self.create_attention_model_before()
            csv_logger = CSVLogger('{}_attention_before.log'.format(self.cnn_model_name), append=True,
                                   separator=';')
        else:
            model = self.create_model_no_attention()
            csv_logger = CSVLogger('{}_no_attention.log'.format(self.cnn_model_name), append=True, separator=';')

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=None)

        train_gen = generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=NUM_EPOCHS,
                                      verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[csv_logger])
        accu = history.history['val_acc'][-1]
        print('cnn-{}, attention-{}: accuracy-{}, time={}'.format(self.cnn_model_name, attention, accu, time.time()-t1))

        return accu
