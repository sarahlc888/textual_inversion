embedding_path = 'logs/woman_occupation32022-12-07T05-57-52_test_run2/checkpoints/embeddings_gs-6099.pt'
ckpt_path = 'models/ldm/text2img-large/model.ckpt'
base_word_path = 'concept_imgs/diverse_occupation_train/base_words.txt'

import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

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


config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
model = load_model_from_config(config, ckpt_path)  # TODO: check path
model.embedding_manager.load(embedding_path)

with open(base_word_path, 'r') as fh:
    for line in fh.readlines():
        bw = line.split()[1]
        model.embedding_manager.get_token_for_string(bw)