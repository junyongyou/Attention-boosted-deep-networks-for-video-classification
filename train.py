import sys
import argparse
import os

from video_classification.generator.attention_cnn_lstm_classifer import BidirectionalLSTMVideoClassifier


def check_args(args):
    if not os.path.exists(args.input_path):
        print('Input path {} does not exist, please check.')
        exit(1)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple script for attention integrated CNN + LSTM video classification training')
    parser.add_argument('cnn_model', help='Specify which CNN model is used (VGG16/VGG19/InceptionV3/Resnet50/Xception')
    parser.add_argument('--attention_mode', help='Specify how to add the attention block (after LSTM: cnn-lstm-attention, before LSTM: cnn-attention-lstm; no attention: cnn-lstm', default='cnn-lstm-attention')
    parser.add_argument('input_path', help='Specify the input data folder path')
    parser.add_argument('--dataset_name', help='Specify the dataset name (UCF-101/Sports-1M)', default='UCF-101')
    parser.add_argument('output_path', help='Specify the output path')
    parser.add_argument('--feature_extraction', help='Specify whether or not do feature extraction first', default=False)
    return check_args(parser.parse_args())


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    classifier = BidirectionalLSTMVideoClassifier(args.cnn_model)
    classifier.fit(input_path=args.input_path, output_path=args.output_path, data_set_name=args.dataset_name, attention=args.attention_mode, do_feature_extraction=args.feature_extraction)


if __name__ == '__main__':
    main()

