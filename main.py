import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

from src_v2.utils import set_seed
from src_v2.datasets import VQADataset
from src_v2.excute import train, eval
from src_v2.models import VQAModel


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transformerモデルの選択
    pretrained_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    # クラスマッピングの読み込み
    class_mapping = pd.read_csv('class_mapping.csv')
    answer_space = class_mapping['answer'].tolist()
    answer2idx = {answer: idx for idx, answer in enumerate(answer_space)}
    idx2answer = {idx: answer for answer, idx in answer2idx.items()}

    # データ拡張の定義
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=train_transform, tokenizer=tokenizer, answer_space=answer_space, answer2idx=answer2idx)
    
    # Split into train and validation
    train_indices, val_indices = train_test_split(range(len(full_train_dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
    
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=test_transform, answer=False, tokenizer=tokenizer, answer_space=answer_space, answer2idx=answer2idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(n_answer=len(answer_space), pretrained_model_name=pretrained_model_name).to(device)

    # optimizer / criterion
    num_epoch = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    best_model = None

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_simple_acc, val_time = eval(model, val_loader, criterion, device)
        
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s], val time: {val_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}, val loss: {val_loss:.4f}\n"
              f"train acc: {train_acc:.4f}, val acc: {val_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}, val simple acc: {val_simple_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load best model
    model.load_state_dict(best_model)

    # 提出用ファイルの作成
    model.eval()
    submission = []
    with torch.no_grad():
        for image, question in test_loader:
            image = image.to(device)
            pred = model(image, question)
            pred = pred.argmax(1).cpu().item()
            submission.append(idx2answer[pred])

    submission = np.array(submission)
    torch.save(model.state_dict(), "model_v2.pth")
    np.save("submission_v2.npy", submission)

if __name__ == "__main__":
    main()