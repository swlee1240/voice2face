import os.path as osp
import random

import torch
from torch.utils.data import Dataset
import numpy as np

import cv2
from PIL import Image
import torchvision.transforms as T
import torchaudio

from dataset import audio_pipeline


class LRS3(Dataset):
    def __init__(
        self,
        root="/home/sangwon/data/lrs3",
        split="train",
        return_types=["audio", "image"],
        input_freq=16_000,
        n_fft=1024,
        hop_length=256,
        n_mel=256,
        f_min=0,
        f_max=8000,
        spec_len=256,
        augs={},
    ):
        super().__init__()

        self.split = split
        self.video_dir = osp.join(root, split, "mp4")
        self.audio_dir = osp.join(root, split, "wav")

        self.return_types = return_types

        self.sr = input_freq

        with open(osp.join(root, split, "file_list.txt"), "r") as f:
            self.flist = f.read().splitlines()

        self.image_pipeline = T.Compose([T.RandomHorizontalFlip(0.5), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
        self.audio_pipeline = audio_pipeline.Pipeline(
            input_freq, n_fft, hop_length, n_mel, f_min, f_max, spec_len, augs=augs
        )

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, index):
        name = self.flist[index]
        video_path = osp.join(self.video_dir, name + ".mp4")
        audio_path = osp.join(self.audio_dir, name + ".wav")

        if "audio" in self.return_types:
            audio, sr = torchaudio.load(audio_path)
            assert sr == self.sr, "sampling rate should be 16k"

            minimum_len = int(sr * 4.5)  # sr * 4 (sec)
            if audio.shape[-1] < minimum_len:
                pad = torch.zeros((1, minimum_len - audio.shape[-1]))
                audio = torch.cat((audio, pad), dim=1)

            audio = self.audio_pipeline(audio)
        else:
            audio = torch.tensor([0.0], dtype=torch.float32)

        if "image" in self.return_types:
            img = self.load_image(video_path)
        else:
            img = torch.tensor([0.0], dtype=torch.float32)

        return {"image": img, "audio": audio, "audio_path":audio_path}

    def load_image(self, path):
        cap = cv2.VideoCapture(path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        rnd_frame = random.randint(0, nframes - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, rnd_frame)

        load_flag, img = cap.read()
        if load_flag is False:
            return None

        cap.release()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, "RGB")

        return self.image_pipeline(img)