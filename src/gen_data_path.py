import os
import glob
import _init_paths


def gen_caltech_path(root_path):
    label_path = 'Caltech/data/labels_with_ids'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.png'))
    with open('../src/data/caltech.all', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.png')
            if image in images_exist:
                print(image[22:], file=f)
    f.close()


def gen_data_path(root_path):
    mot_path = 'MOT17/images/train'

    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('/home/yfzhang/PycharmProjects/fairmot/src/data/mot17.half', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half):
                image = images[i]
                print(image[22:], file=f)
    f.close()

def gen_data_path_sportsmot_train(root_path):
    mot_path = 'sportsmot_mini/images/train'

    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) ]
    with open('/home/fc/PycharmProjects/FairMOT/src/data/sportsmot_mini_for_test.train', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_all):
                image = images[i]
                print(image[18:], file=f)#14 root_path长度加1（加上后面的/）
    f.close()

def gen_data_path_sportsmot_divide_train(root_path):
    mot_path = 'sportsmot_15/images/train'

    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) ]
    with open('/home/fc/PycharmProjects/FairMOT/src/data/sportsmot_divide_15.train', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_all):
                image = images[i]
                print(image[35:], file=f)#14 root_path长度加1（加上后面的/）
    f.close()
def gen_data_path_sportsmot_val(root_path):
    mot_path = 'sportsmot/images/val'

    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) ]
    with open('/home/fc/PycharmProjects/FairMOT/src/data/sportsmot.val', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_all):
                image = images[i]
                print(image[18:], file=f)#14 root_path长度加1（加上后面的/）
    f.close()

def gen_data_path_mot17_val(root_path):
    mot_path = 'MOT17/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('/home/yfzhang/PycharmProjects/fairmot2/src/data/mot17.val', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all):
                image = images[i]
                print(image[22:], file=f)
    f.close()


def gen_data_path_mot17_emb(root_path):
    mot_path = 'MOT17/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
    with open('/home/yfzhang/PycharmProjects/fairmot2/src/data/mot17.emb', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.jpg'))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all, 3):
                image = images[i]
                print(image[22:], file=f)
    f.close()


if __name__ == '__main__':
    # root = '/data/yfzhang/MOT/JDE'
    # gen_data_path_mot17_emb(root)
    root = '/home/fc/datasets'
    gen_data_path_sportsmot_train(root)
