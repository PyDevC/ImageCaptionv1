import os
import pickle
import torch
import nltk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class Vocabulary:
    def __init__(
        self,
        freq_threshold,
        vocab_file="./vocab.pkl",
        start_word="<SOS>",
        end_word="<EOS>",
        unk_word="<UNK>",
        pad_word="<PAD>",
        vocab_from_file=False
    ):
        self.freq_threshold = freq_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.vocab_from_file = vocab_from_file
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def get_vocab(self, captions=None):
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
            print("Vocabulary loaded from file.")
        else:
            self.build_vocabulary(captions)
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocabulary(self, sentence_list):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        for word in [self.pad_word, self.start_word, self.end_word, self.unk_word]:
            self.add_word(word)

        frequencies = Counter()
        for sentence in sentence_list:
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            frequencies.update(tokens)

        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def numericalize(self, text):
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())
        return [
            self.word2idx.get(token, self.word2idx[self.unk_word]) 
            for token in tokenized_text
        ]

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_word])

    def __len__(self):
        return len(self.word2idx)

class CocoDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        ann_file, 
        transform=None, 
        freq_threshold=5, 
        vocab_file="./vocab.pkl", 
        vocab_from_file=False
    ):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        
        self.vocab = Vocabulary(
            freq_threshold=freq_threshold, 
            vocab_file=vocab_file, 
            vocab_from_file=vocab_from_file
        )
        
        if not vocab_from_file:
            all_captions = [ann['caption'] for ann in self.coco.anns.values()]
            self.vocab.get_vocab(all_captions)
        else:
            self.vocab.get_vocab()

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        caption = self.coco.loadAnns(ann_ids)[0]['caption']
        
        img_metadata = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_metadata['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.word2idx[self.vocab.start_word]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx[self.vocab.end_word])

        return image, torch.tensor(numericalized_caption), img_id

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
