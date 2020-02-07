from keras import backend as K
from keras.layers import Input, Dense, Flatten, Activation, Dropout, Bidirectional, Permute, multiply
from keras.layers.recurrent import LSTM
from keras.callbacks import CSVLogger
from keras.models import Sequential, Model, load_model
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
from video_classification.generator.cnn_feature_extractor import CNN_Feature_extractor

K.set_image_dim_ordering('tf')

NUM_EPOCHS = 60
VERBOSE = 1
HIDDEN_UNITS = 256


def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def generate_batch(x_samples, y_samples, batch_size, expected_frames):
    num_batches = len(x_samples) // batch_size

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            x_data = []
            for k in range(start, end):
                x = np.load(x_samples[k])
                frames = x.shape[0]
                if frames > expected_frames:
                    x = x[0:expected_frames, :]
                    x_data.append(x)
                elif frames < expected_frames:
                    temp = np.zeros(shape=(expected_frames, x.shape[1]))
                    temp[0:frames, :] = x
                    x_data.append(temp)
                else:
                    x_data.append(x)

            yield np.array(x_data), y_samples[start:end]


class BidirectionalLSTMVideoClassifier(object):
    def __init__(self, cnn_model_name, model_file=None):
        self.num_input_tokens = None
        self.nb_classes = None
        if model_file is None:
            self.model = None
        else:
            self.model = load_model(model_file)
        self.cnn_model_name = cnn_model_name.lower()

    def cnn_lstm(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=False),
                                input_shape=(self.expected_frames, self.num_input_tokens)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def cnn_attention_lstm(self):
        inputs = Input(shape=(self.expected_frames, self.num_input_tokens,))
        attention_inputs = attention_block(inputs, self.expected_frames)
        lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=False))(attention_inputs)
        x = Dense(512, activation='relu')(lstm_out)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(input=[inputs], output=x)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        return model

    def cnn_lstm_attention(self):
        inputs = Input(shape=(self.expected_frames, self.num_input_tokens,))
        lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
        attention_mul = attention_block(lstm_out, self.expected_frames)
        attention_mul = Flatten()(attention_mul)
        x = Dense(512, activation='relu')(attention_mul)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(input=[inputs], output=x)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        return model

    def predict(self, video_path, config_file):
        config = np.load(config_file).item()
        if self.cnn_model_name == 'vgg16':
            cnn_model = VGG16(include_top=False, weights='imagenet')
        elif self.cnn_model_name == 'vgg19':
            cnn_model = VGG19(include_top=False, weights='imagenet')
        elif self.cnn_model_name == 'inceptionv3':
            cnn_model = InceptionV3(include_top=False, weights='imagenet')
        elif self.cnn_model_name == 'resnet50':
            cnn_model = ResNet50(include_top=False, weights='imagenet')
        else:
            cnn_model = Xception(include_top=False, weights='imagenet')

        feature_extractor = CNN_Feature_extractor(self.cnn_model_name)
        x = feature_extractor.extract_cnn_features_live(cnn_model, video_path)
        expected_frames = config['expected_frames']
        labels_idx2word = dict([(idx, word) for word, idx in config['labels'].items()])
        frames = x.shape[0]
        if frames > expected_frames:
            x = x[0 : expected_frames, :]
        elif frames < expected_frames:
            temp = np.zeros(shape=(expected_frames, x.shape[1]))
            temp[0 : frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = labels_idx2word[predicted_class]
        return predicted_label

    def fit(self, input_path, output_path, data_set_name='UCF-101', test_size=0.3,
            attention='cnn_lstm', do_feature_extraction=False):
        batch_size = 128
        if self.cnn_model_name == 'vgg16':
            print('vgg16')
            cnn_model = VGG16(include_top=False, weights='imagenet')
        elif self.cnn_model_name == 'vgg19':
            print('vgg19')
            cnn_model = VGG19(include_top=False, weights='imagenet')
        elif self.cnn_model_name == 'inceptionv3':
            print('inceptionv3')
            cnn_model = InceptionV3(include_top=False, weights='imagenet')
        elif self.cnn_model_name == 'resnet50':
            print('resnet50')
            cnn_model = ResNet50(include_top=False, weights='imagenet')
            batch_size = 16
        else:
            print('xception')
            cnn_model = Xception(include_top=False, weights='imagenet')
            batch_size = 16

        cnn_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        feature_dir_name = data_set_name + '-' + self.cnn_model_name + '-HiDimFeatures'

        labels = dict()
        feature_extractor = CNN_Feature_extractor(self.cnn_model_name)
        x_samples, y_samples = feature_extractor.extract_cnn_features(input_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=cnn_model,
                                                               data_set_name=data_set_name)
        if do_feature_extraction:
            return
        x_sample_0 = np.load(x_samples[0])
        print(x_sample_0.shape)
        self.num_input_tokens = x_sample_0.shape[1]
        frames_list = []
        for x in x_samples:
            frames = np.load(x).shape[0]
            frames_list.append(frames)
        self.expected_frames = int(np.mean(frames_list))
        print('expected frames: ', self.expected_frames)

        for y in y_samples:
            if y not in labels:
                labels[y] = len(labels)

        for i in range(len(y_samples)):
            y_samples[i] = labels[y_samples[i]]

        self.nb_classes = len(labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames

        np.save(os.path.join(output_path, 'config'), config)

        t1 = time.time()
        if attention == 'cnn_lstm_attention':
            model = self.cnn_lstm_attention()
            csv_logger = CSVLogger(os.path.join(output_path, '{}_cnn_lstm_attention.log'.format(self.cnn_model_name)), append=True, separator=';')
        elif attention == 'cnn_attention_lstm':
            model = self.cnn_attention_lstm()
            csv_logger = CSVLogger(os.path.join(output_path, '{}_cnn_attention_lstm.log'.format(self.cnn_model_name)), append=True,
                                   separator=';')
        else:
            model = self.cnn_lstm()
            csv_logger = CSVLogger(os.path.join(output_path, '{}_cnn_lstm.log'.format(self.cnn_model_name)), append=True, separator=';')

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=None)

        train_gen = generate_batch(Xtrain, Ytrain, batch_size, self.expected_frames)
        test_gen = generate_batch(Xtest, Ytest, batch_size, self.expected_frames)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=NUM_EPOCHS,
                                      verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[csv_logger])

        model_file_path = os.path.join(output_path, attention.replace('cnn', self.cnn_model_name) + '.h5')
        model.save(model_file_path)
        accu = history.history['val_acc'][-1]
        print('cnn-{}, attention-{}: accuracy-{}, time={}'.format(self.cnn_model_name, attention, accu, time.time()-t1))

        return accu
