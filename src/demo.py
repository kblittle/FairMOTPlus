from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    # dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    dataloader = datasets.LoadImages('/home/fc/datasets/sportsmot/images/val/v_0kUtTtmLaJA_c010/img1', opt.img_size)
    # v_9MHDmAMxO5I_c004-volley
    # v_ITo3sCnpw_k_c007-foot
    # v_5ekaksddqrc_c003-basket
    # basketball    v_5ekaksddqrc_c001 first
    # football     v_ITo3sCnpw_k_c010 first
    # volleyball    v_0kUtTtmLaJA_c010 first



    result_filename = os.path.join(result_root, 'results.txt')
    # frame_rate = dataloader.frame_rate
    frame_rate = 25

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    #删除上一次算法跟踪的结果
    del_list = os.listdir(frame_dir)
    for f in del_list:
        file_path = os.path.join(frame_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        # output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        output_video_path = osp.join(result_root,'ch_sportsmot_hrnet18_byte_epoch60_e15' ,'v_0kUtTtmLaJA_c010-volley-first01.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
