
import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p, get_rank, merge_args_yaml, get_time_stamp, load_model_opt, save_args,load_npz
from lib.utils import tokenize, truncated_noise, prepare_sample_data,load_netG, encode_clip_tokens,get_clip_caption
from lib.perpare import prepare_datasets, prepare_models,prepare_dataloaders
from lib.modules import sample_one_batch as sample, test as test, train as train
#from lib.datasets_clip import get_fix_data
from lib.datasets import get_fix_data,prepare_data,get_imgs
import pickle


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DE-Net')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/bird.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers(default: 1)')
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--imsize', type=int, default=256,
                        help='image size')
    parser.add_argument('--model', type=str, default='model0',
                        help='the model for training')
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='state epoch')
    parser.add_argument('--resume_model_path', type=str, default='model',
                        help='the filepath of saved checkpoints to resume')
    parser.add_argument('--clip_path', type=str, default='RN50',
                        help='clip path')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--imgs_per_sent', type=int, default=10,
                        help='imgs_per_sent')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True,
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--state_epoch', type=int, default=1,
                        help='state epoch')
    parser.add_argument('--pretrained_model_path', type=str, default='model',
                        help='the model for training')
    parser.add_argument('--log_dir', type=str, default='new',
                        help='file path to log directory')
    args = parser.parse_args()
    return args

def build_word_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
        wordtoix = x[3]
        del x
        n_words = len(wordtoix)
        print('Load from: ', pickle_path)
    return n_words, wordtoix

def sample_example(wordtoix, netG, text_encoder, args):
    batch_size, device = args.TEXT.CAPTIONS_PER_IMAGE, args.device
    batch_size = 1
    text_filepath, img_filepath, img_save_path = args.example_captions, args.example_images, args.samples_save_dir

    # truncation, trunc_rate = args.truncation, args.trunc_rate
    # z_dim = args.z_dim

    #captions, cap_lens, _ = tokenize(wordtoix, text_filepath)
    #sent_embs, _  = prepare_sample_data(captions, cap_lens, text_encoder, device)
    #caption_num = sent_embs.size(0)

    caps_clip, clip_tokens = get_clip_caption(text_filepath, args.clip4text)
    clip_tokens = clip_tokens.to(device)
    sent_embs, words_embs = encode_clip_tokens(text_encoder, clip_tokens)
    caption_num = sent_embs.size(0)

    # # get noise
    # if truncation==True:
    #     noise = truncated_noise(batch_size, z_dim, trunc_rate)
    #     noise = torch.tensor(noise, dtype=torch.float).to(device)
    # else:
    #     noise = torch.randn(batch_size, z_dim).to(device)

    #get image
    image_transform = transforms.Compose([
            transforms.Resize((args.imsize, args.imsize))
            #transforms.Resize(int(args.imsiz,e * 76 / 64)),
            #transforms.RandomCrop(args.imsize),
            #transforms.RandomHorizontalFlip(),
            ])
    image_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    image = get_imgs(img_filepath, bbox=None, transform=image_transform, normalize=image_norm)
    image = image.to(device)

    # sampling
    with torch.no_grad():
        fakes = []
        for i in tqdm(range(caption_num)):
            #sent_emb = sent_embs[i].unsqueeze(0).repeat(batch_size, 1)
            sent_emb = sent_embs[i].unsqueeze(0)
            image_input = image.unsqueeze(0)
            #fakes = netG(noise, sent_emb)

            noise = torch.randn(batch_size, 100).to(device)
            if (args.multi_gpus == True):
                netG.module.lstm.init_hidden(noise)
            else:
                netG.lstm.init_hidden(noise)

            fakes = netG(image_input, sent_emb)
            img_name = osp.join(img_save_path,'Sent%03d.png'%(i+1))
            vutils.save_image(fakes.data, img_name, nrow=4, range=(-1, 1), normalize=True)
            torch.cuda.empty_cache()

def gen_example(wordtoix, netG, text_encoder, args):
    batch_size, device = args.TEXT.CAPTIONS_PER_IMAGE, args.device
    batch_size = 1
    #text_filepath, img_filepath, img_save_path = args.example_captions, args.example_images, args.samples_save_dir

    filepath = '%s/example_filenames.txt' % (args.data_dir)
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue

            #img_name = name.replace("textchange", "images") #cub
            #img_filepath = '%s/CUB_200_2011/%s.jpg' % (args.data_dir, img_name)  #cub

            img_name = name.replace("textchange", "images_val") #coco
            #img_name = name.replace("textchange", "images") #coco train
            img_filepath = '%s/%s.jpg' % (args.data_dir, img_name)  #coco

            # get image
            image_transform = transforms.Compose([
                transforms.Resize((args.imsize, args.imsize))
                # transforms.Resize(int(args.imsiz,e * 76 / 64)),
                # transforms.RandomCrop(args.imsize),
                # transforms.RandomHorizontalFlip(),
            ])
            image_norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            image = get_imgs(img_filepath, bbox=None, transform=image_transform, normalize=image_norm)
            image = image.to(device)

            text_filepath = '%s/%s.txt' % (args.data_dir, name)

            caps_clip, clip_tokens = get_clip_caption(text_filepath, args.clip4text)
            clip_tokens = clip_tokens.to(device)
            sent_embs, words_embs = encode_clip_tokens(text_encoder, clip_tokens)
            caption_num = sent_embs.size(0)

            key = name[(name.rfind('/') + 1):]

            # sampling
            with torch.no_grad():
                fakes = []
                for i in tqdm(range(caption_num)):
                    # sent_emb = sent_embs[i].unsqueeze(0).repeat(batch_size, 1)
                    sent_emb = sent_embs[i].unsqueeze(0)
                    image_input = image.unsqueeze(0)
                    # fakes = netG(noise, sent_emb)

                    noise = torch.randn(batch_size, 100).to(device)
                    if (args.multi_gpus == True):
                        netG.module.lstm.init_hidden(noise)
                    else:
                        netG.lstm.init_hidden(noise)

                    fakes = netG(image_input, sent_emb)
                    img_name = osp.join(args.samples_save_dir, '%sSent%03d.png' % (key, (i + 1)))
                    vutils.save_image(fakes.data, img_name, nrow=4, range=(-1, 1), normalize=True)
                    torch.cuda.empty_cache()

def main(args):
    time_stamp = get_time_stamp()
    args.samples_save_dir = osp.join(args.samples_save_dir, time_stamp)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(args.samples_save_dir)
    # prepare data
    #pickle_path = os.path.join(args.data_dir, 'captions_DAMSM.pickle')
    pickle_path = os.path.join(args.data_dir, 'captions.pickle')
    args.n_words, wordtoix = build_word_dict(pickle_path)
    # prepare models
    _, text_encoder, _, _, netG, _, _, _ = prepare_models(args)
    #model_path = osp.join(ROOT_PATH, args.checkpoint)
    #model_path = args.checkpoint
    model_path = args.pretrained_model_path

    netG = load_netG(netG, model_path, args.multi_gpus, train=True)
    netG.eval()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('Load %s for NetG'%(args.checkpoint))
        print("************ Start sampling ************")
    start_t = time.time()
    #sample_example(wordtoix, netG, text_encoder, args)
    gen_example(wordtoix, netG, text_encoder, args)
    end_t = time.time()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('*'*40)
        print('Sampling done, %.2fs cost, saved to %s'%(end_t-start_t, args.samples_save_dir))
        print('*'*40)



if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)
