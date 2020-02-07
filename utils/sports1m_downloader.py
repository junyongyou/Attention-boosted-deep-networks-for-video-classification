import json
import os
from pytube import YouTube
import collections
import numpy as np
import shutil


youtube_link ='https://www.youtube.com/watch?v='


def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)

    # print('Done reading {:s}: {:d} items'.format(json_file, len(data)))
    return data


def check_category_info(json_data, max_len=10000000, min_len=-1):
    count = 0
    categories = dict()

    for i, datum in enumerate(json_data):
        for label in datum['label487']:
            if label in categories:
                num = categories[label]
                num += 1
                categories[label] = num
            else:
                categories[label] = 1

    categories = collections.OrderedDict(sorted(categories.items()))
    for category in categories:
        print('{}: {}'.format(category, categories[category]))

    return categories


def check_info(json_data, train_set=None, max_len=10, min_len=6):
    count = 0
    selected_categories = set()

    for i, datum in enumerate(json_data):
        if datum['duration'] >= min_len and datum['duration'] <= max_len:
            videolabel = ''
            for label in datum['label487']:
                videolabel += str(label)
                try:
                    if train_set:
                        if label in train_set:
                            selected_categories.add(label)
                    else:
                        selected_categories.add(label)
                except Exception:
                    t = 0
            count += 1

    print('{} samples'.format(count))
    return selected_categories


def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def download(json_data, save_dir, sample_set, max_len=10, min_len=6, dstfile=None):
    save_dir = get_dir(save_dir)
    count = 0
    nfiles =len(json_data)

    with open(dstfile, 'w') as f:
        for i, datum in enumerate(json_data):
            if datum['duration'] >= min_len and datum['duration'] <= max_len:
                videolink = youtube_link + datum['id']
                try:
                    for label in datum['label487']:
                        if label in sample_set:
                            videoid = datum['id']
                            video = YouTube(videolink).streams.first()
                            video_name = os.path.join(save_dir, videoid + ' ' + str(label) + '.' + video.subtype)
                            if os.path.exists(video_name):
                                print('Video downloaded already, skip')
                                continue
                            video.download(save_dir)
                            os.rename(os.path.join(save_dir, video.default_filename), video_name)

                            # vidcap = cv2.VideoCapture(video_name)
                            count += 1
                            print('{:d} : {:d} -- {:s}, len: {:d}'.format(i, nfiles, datum['id'], int(datum['duration'])))
                except Exception as e:
                    print(e)
                    continue

    print('{}'.format(count))
    # print('Done Download, {:d} is downloaded, information saved to '.format(count))


def download_202class(json_data, save_dir, dstfile=None, max_len=20):
    save_dir = get_dir(save_dir)
    # count = 0
    nfiles =len(json_data)
    counter = np.zeros(101)

    with open(dstfile, 'w') as f:
        for i, datum in enumerate(json_data):
            if i >= int(nfiles / 2):
                continue
            videolink = youtube_link + datum['id']
            try:
                if datum['duration'] >= max_len:
                    continue
                for label in datum['label487']:
                    if int(label) < 202:
                        sub_dir = os.path.join(save_dir, str(label))
                        if not os.path.exists(sub_dir):
                            os.makedirs(sub_dir)

                        if counter[int(label)] >= 300:
                            continue
                        videoid = datum['id']
                        video = YouTube(videolink).streams.first()
                        existing_video_file = os.path.join(save_dir, videoid + ' ' + str(label) + '.' + video.subtype)

                        video_name = os.path.join(sub_dir, videoid + ' ' + str(label) + '.' + video.subtype)
                        if os.path.exists(existing_video_file):
                            print('Video downloaded already, {} moved'.format(existing_video_file))
                            shutil.move(existing_video_file, video_name)
                            continue
                        if os.path.exists(video_name):
                            print('Video {} downloaded already'.format(video_name))
                            continue

                        video.download(sub_dir)
                        os.rename(os.path.join(sub_dir, video.default_filename), video_name)

                        counter[int(label) - 101] += 1
                        print('{:d} : {:d} -- {:s}, len: {:d}'.format(i, nfiles, datum['id'], int(datum['duration'])))
            except Exception as e:
                print(e)
                continue


def main(srcfile, savedir, sample_set=None):
    dstfile = None
    savedir = get_dir(savedir)

    raw_json_data = read_json(srcfile)
    if not dstfile:
        srcname, srcext = os.path.splitext(srcfile)
        dstfile = '{:s}-{:02d}-{:02d}{:s}'.format(srcname, 5, 10, '.txt')
    download_202class(raw_json_data, savedir, dstfile=dstfile)
    # download(raw_json_data, save_dir=savedir, sample_set=sample_set, dstfile=dstfile)


if __name__ == '__main__':
    # vgg19_model = VGG19(include_top=False, weights='imagenet')
    # vgg19_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    srcfile_train = r'C:\Users\junyong\Downloads\sports1m_json\sports1m_train.json'
    # srcfile_test = r'C:\Users\junyong\Downloads\sports1m_json\sports1m_test.json'

    # train_categories = check_info(read_json(srcfile_train))
    # test_categories = check_info(read_json(srcfile_test), train_categories)

    # check_category_info(read_json(srcfile_train))

    savedir_train = r'C:\Users\junyong\Downloads\sports_1m\train'
    main(srcfile_train, savedir_train)
    # main(srcfile_train, savedir_train, train_categories)

    # savedir_test = r'C:\Users\junyong\Downloads\sports_1m\test'
    # main(srcfile_test, savedir_test, test_categories)

    # print('{} train categories, {} test categories'.format(len(train_categories), len(test_categories)))
    # if test_categories.issubset(train_categories):
    #     print('Train covers test')
    # else:
    #     print('Train not covers test')