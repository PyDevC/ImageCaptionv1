import os
import torch
import nltk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in nltk.tokenize.word_tokenize(sentence.lower()):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, freq_threshold=5):
        """
        Args:
            root_dir: path to the folder containing images (e.g., "data/COCO/images/train2017")
            ann_file: path to the captions json file
        """
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        
        self.vocab = Vocabulary(freq_threshold)
        all_caption_dicts = self.coco.anns.values()
        captions = [d['caption'] for d in all_caption_dicts]
        self.vocab.build_vocabulary(captions)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        caption = self.coco.loadAnns(ann_ids)[0]['caption']
        
        img_metadata = self.coco.loadImgs(img_id)[0]
        img_filename = img_metadata['file_name']
        
        img_path = os.path.join(self.root_dir, img_filename)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)

    def __len__(self):
        return len(self.ids)

class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets
