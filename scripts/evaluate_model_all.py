import argparse, os, sys, glob

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.personalized import PersonalizedBase
from evaluation.clip_eval import LDMCLIPEvaluator

import pandas as pd 

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def eval_model(model, base_word, data_dir, prompt="a photo of *"):

    data_loader = PersonalizedBase(data_dir, size=256, flip_p=0.0)

    images = [torch.from_numpy(data_loader[i]["image"]).permute(2, 0, 1) for i in range(data_loader.num_images)]
    images = torch.stack(images, axis=0)

    sim_img, sim_text = evaluator.evaluate(model, images, prompt, base_word=base_word)

    return sim_img, sim_text


if __name__ == '__main__':
    ckpt_path = 'models/ldm/text2img-large/model.ckpt'
    eval_data_root = 'concept_imgs/woman_occ3_by_occ'
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-finetune-DELTA.yaml")  

    base_words = os.listdir(eval_data_root) 
    
    emb_paths = [
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-499.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-999.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-1499.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-1999.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-2499.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-2999.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-3499.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-3999.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-4499.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-4999.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-5499.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-5999.pt',
        'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-6099.pt',
    ]
    
    results_file = 'eval.out.txt'

    df = pd.read_csv(results_file, sep='\t', header=None, names=['emb_file', 'base_word', 'sim_img'])

    model = load_model_from_config(config, ckpt_path) 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    for embedding_path in emb_paths:

        model.embedding_manager.load(embedding_path)

        evaluator = LDMCLIPEvaluator(device)

        for base_word in base_words:
            if base_word in df[ df['emb_file'] == embedding_path ]['base_word'].values:
                print("SKIPPING", base_word, embedding_path)
                continue 

            print("PROCESSING"), base_word
            base_word_data_dir = os.path.join(eval_data_root, base_word) 

            sim_img, sim_text = eval_model(model, base_word, base_word_data_dir)

            # add a row and write to output
            df.loc[len(df.index)] = [embedding_path, base_word, sim_img.cpu().item()] 
            df.to_csv(results_file, sep='\t', index=False)