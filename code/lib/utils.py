import os
import sys
import errno
import numpy as np
import numpy.random as random
import torch
from torch import distributed as dist
import json
import pickle
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
import pprint
import datetime
import dateutil.tz
from io import BytesIO
from PIL import Image
from torchvision import transforms, datasets
import torch.nn.functional as F
import models.model as basic_model
import torch.nn as nn
import math
import clip

def choose_model(model):
    NetG = basic_model.NetG
    NetD = basic_model.NetD
    NetC = basic_model.NetC
    CLIP_IMG_ENCODER = basic_model.CLIP_IMG_ENCODER
    CLIP_TXT_ENCODER = basic_model.CLIP_TXT_ENCODER
    NetD_DINO = basic_model.NetD_DINO

    return NetG, NetD, NetC, CLIP_IMG_ENCODER, CLIP_TXT_ENCODER, NetD_DINO

# test_utils
def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def load_npz(path):
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s


def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# config
def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp


def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        args.update(opt)
        args = edict(args)
    return args


def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()


# DDP utils
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


# save and load models
def load_opt_weights(optimizer, weights):
    optimizer.load_state_dict(weights)
    return optimizer


def load_model_opt(netG, netD, netC, netD_DINO, optim_G, optim_D, path, multi_gpus):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus)
    netD = load_model_weights(netD, checkpoint['model']['netD'], multi_gpus)
    netC = load_model_weights(netC, checkpoint['model']['netC'], multi_gpus)
    netD_DINO = load_model_weights(netD_DINO, checkpoint['model']['netD_DINO'], multi_gpus)
    optim_G = load_opt_weights(optim_G, checkpoint['optimizers']['optimizer_G'])
    optim_D = load_opt_weights(optim_D, checkpoint['optimizers']['optimizer_D'])
    return netG, netD, netC, netD_DINO, optim_G, optim_D


def load_models(netG, netD, netC, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'])
    netD = load_model_weights(netD, checkpoint['model']['netD'])
    netC = load_model_weights(netC, checkpoint['model']['netC'])
    return netG, netD, netC


def load_netG(netG, path, multi_gpus, train):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus, train)
    return netG


def load_model_weights(model, weights, multi_gpus, train=True):
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model


def save_models(netG, netD, netC, optG, optD, epoch, multi_gpus, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))


# data util
def write_to_txt(filename, contents): 
    fh = open(filename, 'w') 
    fh.write(contents) 
    fh.close()


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


###########  GEN  #############
def get_tokenizer():
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer


def tokenize(wordtoix, text_filepath):
    '''generate images from example sentences'''
    tokenizer = get_tokenizer()
    filepath = text_filepath
    with open(filepath, "r") as f:
        sentences = f.read().split('\n')
        # a list of indices for a sentence
        captions = []
        cap_lens = []
        new_sent = []
        for sent in sentences:
            if len(sent) == 0:
                continue
            sent = sent.replace("\ufffd\ufffd", " ")
            tokens = tokenizer.tokenize(sent.lower())
            if len(tokens) == 0:
                print('sent', sent)
                continue
            rev = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in wordtoix:
                    rev.append(wordtoix[t])
            captions.append(rev)
            cap_lens.append(len(rev))
            new_sent.append(sent)
        return captions, cap_lens, new_sent


def sort_example_captions(captions, cap_lens, device):
    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    captions = torch.from_numpy(cap_array).to(device)
    cap_lens = torch.from_numpy(cap_lens).to(device)
    return captions, cap_lens, sorted_indices


def prepare_sample_data(captions, caption_lens, text_encoder, device):
    print('*'*40)
    captions, sorted_cap_lens, sorted_cap_idxs = sort_example_captions(captions, caption_lens, device)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    return sent_emb, words_embs


def encode_tokens(text_encoder, caption, cap_lens):
    # encode text
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            hidden = text_encoder.init_hidden(caption.size(0))
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs

def get_clip_caption(cap_path,clip_info):
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)

    #sent_ix = random.randint(0, len(eff_captions))
    #caption = eff_captions[sent_ix]
    caption = eff_captions
    tokens = clip.tokenize(caption,truncate=True)
    return caption, tokens

def encode_clip_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb, words_embs = text_encoder(caption)
        sent_emb, words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs

def sort_sents(captions, caption_lens, device):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = captions.to(device)
    sorted_cap_lens = sorted_cap_lens.to(device)
    return captions, sorted_cap_lens, sorted_cap_indices


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def save_img(img, path):
    im = img.data.cpu().numpy()
    # [-1, 1] --> [0, 255]
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)
    im.save(path)

def transf_to_CLIP_input(inputs):
    device = inputs.device
    if len(inputs.size()) != 4:
        raise ValueError('Expect the (B, C, X, Y) tensor.')
    else:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
        inputs = ((inputs+1)*0.5-mean)/var
        return inputs.float()

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

class MogLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, mog_interation):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.mog_interations = mog_interation

        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))

        #Mogrifiers
        self.Q = nn.Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()
        self.noise2h = nn.Linear(100, 512)  # ViT-B/32
        self.noise2c = nn.Linear(100, 512)  # ViT-B/32
        #self.noise2h = nn.Linear(100,868)  #ViT-L/14
        #self.noise2c = nn.Linear(100,868)  #ViT-L/14
        #self.init_hidden()
        self.hidden_seq = []
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def init_hidden(self,noise):
        h_t = self.noise2h(noise)
        c_t = self.noise2c(noise)

        self.c_t = c_t
        self.h_t = h_t

    def mogrify(self, xt, ht):
        for i in range(1, self.mog_interations+1):
            if(i % 2 == 0):
                ht = (2*torch.sigmoid(xt @ self.R)*ht)
            else:
                xt = (2*torch.sigmoid(ht @ self.Q)*xt)
        return xt, ht

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        #bs, seq_sz, _ = x.size()
        #hidden_seq = []
        c_t = self.c_t
        h_t = self.h_t
        HS = self.hidden_size
#        x_t = x[:, t, :]
        x_t = x
        x_t, h_t = self.mogrify(x_t, h_t)

        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]), # input
            torch.sigmoid(gates[:, HS:HS*2]), # forget
            torch.tanh(gates[:, HS*2:HS*3]),
            torch.sigmoid(gates[:, HS*3:]), # output
        )
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        self.c_t = c_t
        self.h_t = h_t

        return h_t, c_t