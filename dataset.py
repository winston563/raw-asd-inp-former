import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import torch.nn.functional as F
import torchaudio

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class AudioAnomalyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        phase,
        sample_rate=16000,
        duration=0.3,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=64,
        f_min=0.0,
        f_max=None,
        input_size=448,
        crop_size=392,
        eps=1e-6,
    ):
        self.root = root
        self.phase = phase
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.input_size = input_size
        self.crop_size = crop_size
        self.eps = eps

        self.target_length = int(round(self.sample_rate * self.duration))
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0,
        )

        if self.phase == "train":
            self.data_root = os.path.join(self.root, "train", "ok")
            self.labels = [0]
        else:
            self.data_root = os.path.join(self.root, "test")
            self.labels = None

        self.audio_paths, self.audio_labels = self._load_audio_paths()

    def _load_audio_paths(self):
        if self.phase == "train":
            audio_paths = sorted(glob.glob(os.path.join(self.data_root, "*.wav")))
            audio_labels = [0] * len(audio_paths)
            return audio_paths, audio_labels

        ok_paths = sorted(glob.glob(os.path.join(self.data_root, "ok", "*.wav")))
        ng_paths = sorted(glob.glob(os.path.join(self.data_root, "ng", "*.wav")))
        audio_paths = ok_paths + ng_paths
        audio_labels = [0] * len(ok_paths) + [1] * len(ng_paths)
        return audio_paths, audio_labels

    def __len__(self):
        return len(self.audio_paths)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if waveform.numel() < self.target_length:
            pad = self.target_length - waveform.numel()
            waveform = F.pad(waveform, (0, pad))
        elif waveform.numel() > self.target_length:
            waveform = waveform[: self.target_length]
        return waveform

    def _to_log_mel(self, waveform):
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel + self.eps)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + self.eps)
        log_mel = log_mel.unsqueeze(0).unsqueeze(0)
        log_mel = F.interpolate(log_mel, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        if self.crop_size and self.crop_size < self.input_size:
            start = (self.input_size - self.crop_size) // 2
            log_mel = log_mel[:, :, start : start + self.crop_size, start : start + self.crop_size]
        log_mel = log_mel.repeat(1, 3, 1, 1).squeeze(0)
        return log_mel

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.audio_labels[idx]
        waveform = self._load_audio(audio_path)
        log_mel = self._to_log_mel(waveform)

        if self.phase == "train":
            return log_mel, label

        if label == 0:
            gt = torch.zeros([1, log_mel.size(-2), log_mel.size(-1)])
        else:
            gt = torch.ones([1, log_mel.size(-2), log_mel.size(-1)])

        return log_mel, gt, label, audio_path

