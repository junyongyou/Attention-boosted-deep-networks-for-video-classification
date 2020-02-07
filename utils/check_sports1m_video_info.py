import os
import glob
import shutil
import cv2


def read_labels():
    label_file = r'D:\Sports-1M\labels.txt'
    f = open(label_file, 'r')
    labels = f.readlines()
    f.close()
    return labels


def copy_folder(target_folder):
    folder = r'D:\Sports-1M\train'
    num = 0
    labels = read_labels()
    for i in range(202):
        sub_folder = os.path.join(folder, str(i))
        files = glob.glob(os.path.join(sub_folder, '*.*'))
        if len(files) > 100:
            print('{}: {}'.format(i, len(files)))
            num += 1
        new_folder = os.path.join(target_folder, labels[i].strip())
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        shutil.copytree(sub_folder, new_folder)
    print(num)


def check_video_frames(target_folder):
    folder = r'D:\Sports-1M\train\*'
    frames = 0
    num_video = 0
    sub_folders = glob.glob(folder)
    for sub_folder in sub_folders:
        files = glob.glob(os.path.join(sub_folder, '*.*'))
        for file in files:
            cap = cv2.VideoCapture(os.path.join(sub_folder, file))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames += length
            num_video += 1
    print('Total videos: {}'.format(num_video))
    print('Total frames: {}'.format(frames))
    print('Average frames: {}'.format(int(frames / num_video)))


if __name__ == '__main__':
    read_labels()
    # target_folder = r'C:\'
    # copy_folder(target_folder)