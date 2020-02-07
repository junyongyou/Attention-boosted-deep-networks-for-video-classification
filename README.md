# Attention-boosted-deep-networks-for-video-classificaton
This is a implementation of integrating a simple but efficient attention block in CNN + bidirectional LSTM for video classification.

# Requirements
Please install the packages listed in requirements.txt. Anaconda + PyCharm are recommended.

# Train the model
Run Python train.py cnn_model attention_mode(optional) input_path dataset_name output_path feature_extraction(optional)

Training a model for the first time, it is recommended to use the feature_extraction argument, which can extract image features by CNN and then store them in npy files.

Please see train.py for details about the training arguments.
```python
def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple script for attention integrated CNN + LSTM video classification training')
    parser.add_argument('cnn_model', help='Specify which CNN model is used (VGG16/VGG19/InceptionV3/Resnet50/Xception')
    parser.add_argument('--attention_mode', help='Specify how to add the attention block (after LSTM: cnn-lstm-attention, before LSTM: cnn-attention-lstm; no attention: cnn-lstm', default='cnn-lstm-attention')
    parser.add_argument('input_path', help='Specify the input data folder path')
    parser.add_argument('--dataset_name', help='Specify the dataset name (UCF-101/Sports-1M)', default='UCF-101')
    parser.add_argument('output_path', help='Specify the output path')
    parser.add_argument('--feature_extraction', help='Specify whether or not do feature extraction first', default=False)
    return check_args(parser.parse_args())
```
After the training is complete, respective config information and model will be stored in the output_path folder, which can be used in prediction of new video class.

# Predict video class
Run Python predict.py cnn_model model_path video_path config_path
```python
def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple script for attention integrated CNN + LSTM video classification')
    parser.add_argument('cnn_model', help='Specify which CNN model is used (VGG16/VGG19/InceptionV3/Resnet50/Xception')
    parser.add_argument('model_path', help='Specify the model path')
    parser.add_argument('video_path', help='Specify the input video path')
    parser.add_argument('config_path', help='Specify the config file path')
    return parser.parse_args(args)
```

# Datasets
Information about two datasets (UCF101 and Sports-1M) is presented in the utils module, including the 99 video classes together with download links from the Sports-1M.  

