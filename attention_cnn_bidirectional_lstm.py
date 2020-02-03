import numpy as np
from keras import backend as K
import os
import gc

from video_classification.cnn_classifer import BidirectionalLSTMVideoClassifier
from video_classification.UCF101_loader import load_ucf

def main():
    # K.set_image_dim_ordering('tf')

    data_set_name = 'UCF-101'
    input_dir_path = r'C:\UCF101'
    output_dir_path = os.path.join(os.path.dirname(__file__), 'models', data_set_name)
    report_dir_path = os.path.join(os.path.dirname(__file__), 'reports', data_set_name)

    # load_ucf(input_dir_path)

    cnn_model_names = ['inceptionv3', 'vgg19', 'vgg16', 'resnet50', 'xception']
    attentions = [1, 2, 0]
    R = 10

    for cnn_model_name in cnn_model_names:
        for attention in attentions:            
            for k in range(R):
                result_file = os.path.join(r'C:\fish_lice_dataset\UCF101', 'cnn_result_' + cnn_model_name +
                                           '_' + str(attention) + '_fold_' + str(k) + '.csv')
                with open(result_file, 'w+') as rf:
                    classifier = BidirectionalLSTMVideoClassifier(cnn_model_name)

                    accuracy = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, include_top=False,
                                                data_set_name=data_set_name, attention=attention)

                    rf.write('*** Result for {}\n'.format(cnn_model_name))
                    rf.write('   Average accuracy: {}\n'.format(accuracy))
                    rf.write('\n')
                gc.collect()


if __name__ == '__main__':
    main()
