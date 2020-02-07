import sys
import argparse
import os

from video_classification.generator.attention_cnn_lstm_classifer import BidirectionalLSTMVideoClassifier


def check_args(args):
    if not os.path.exists(args.model_path):
        print('Model path {} does not exist, please check.')
        exit(1)
    if not os.path.exists(args.video_path):
        print('Video path {} does not exist, please check.')
        exit(1)
    if not os.path.exists(args.config_path):
        print('Config file {} does not exist, please check.')
        exit(1)
    return args


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple script for attention integrated CNN + LSTM video classification')
    parser.add_argument('cnn_model', help='Specify which CNN model is used (VGG16/VGG19/InceptionV3/Resnet50/Xception')
    parser.add_argument('model_path', help='Specify the model path')
    parser.add_argument('video_path', help='Specify the input video path')
    parser.add_argument('config_path', help='Specify the config file path')
    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    classifier = BidirectionalLSTMVideoClassifier(args.cnn_model, args.model_path)
    predicted_label = classifier.predict(args.video_path, args.config_path)
    print('{} belongs to {}'.format(args.video_path, predicted_label))


if __name__ == '__main__':
    main()

