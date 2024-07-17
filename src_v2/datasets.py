import pandas as pd
import torch
from PIL import Image
from Levenshtein import distance

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, tokenizer=None, answer_space=None, answer2idx=None):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer
        self.tokenizer = tokenizer
        self.answer_space = answer_space
        self.answer2idx = answer2idx

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        print(f"DONE: {self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        
        question = self.df["question"][idx]
        
        if self.answer:
            answers = self.df["answers"][idx]
            # 最も頻繁な回答を選択
            mode_answer = max(answers, key=lambda x: x['answer_confidence'])['answer']
            # answer_spaceに存在しない回答の場合、最も類似した回答を選択
            if mode_answer not in self.answer2idx:
                mode_answer = min(self.answer_space, key=lambda x: distance(x.lower(), mode_answer.lower()))
            mode_answer_idx = self.answer2idx[mode_answer]

            # すべての回答のインデックスを取得
            answer_indices = [self.answer2idx.get(a['answer'], self.answer2idx['unanswerable']) for a in answers]

            return image, question, torch.tensor(answer_indices), mode_answer_idx
        else:
            return image, question

    def __len__(self):
        return len(self.df)