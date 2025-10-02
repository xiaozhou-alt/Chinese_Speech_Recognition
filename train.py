import os
import random
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import soundfile as sf
from tqdm.auto import tqdm
from IPython.display import Audio, display
import openpyxl
import evaluate  # 用于计算WER指标

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化混合精度训练的scaler
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
print(f"混合精度训练: {'启用' if device.type == 'cuda' else '禁用'}(需要CUDA支持)")

# 加载WER评估指标
wer_metric = evaluate.load("wer")

# 数据集路径
data_dir = "/kaggle/input/chinese-speech-to-textthchs30/data/data"  # 修改为你的数据路径

# 获取所有文件
wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
print(f"Found {len(wav_files)} WAV files")

# 划分训练集和验证集 (80%训练, 20%验证)
train_size = int(0.8 * len(wav_files))
val_size = len(wav_files) - train_size
train_files, val_files = random_split(wav_files, [train_size, val_size])

print(f"Training set: {len(train_files)} samples")
print(f"Validation set: {len(val_files)} samples")

# 创建自定义数据集类
class SpeechDataset(Dataset):
    def __init__(self, file_list, data_dir):
        self.file_list = file_list
        self.data_dir = data_dir
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        wav_file = self.file_list[idx]
        wav_path = os.path.join(self.data_dir, wav_file)
        trn_path = os.path.join(self.data_dir, wav_file + ".trn")
        
        # 读取音频文件
        try:
            speech_array, sampling_rate = sf.read(wav_path)
            speech_array = speech_array.astype(np.float32)
        except Exception as e:
            print(f"读取音频文件错误 {wav_path}: {e}")
            return None
        
        # 如果音频是双声道，转换为单声道
        if len(speech_array.shape) > 1:
            speech_array = np.mean(speech_array, axis=1)
        
        # 重采样到16kHz
        if sampling_rate != 16000:
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        
        # 读取标签
        try:
            with open(trn_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 使用第一行作为标签文本
                text = lines[0].strip() if lines else ""
        except Exception as e:
            print(f"读取标签文件错误 {trn_path}: {e}")
            text = ""
        
        # 处理音频和标签
        inputs = self.processor(
            speech_array, 
            sampling_rate=16000, 
            text=text,
            padding=True,
            return_tensors="pt"
        )
        
        # 移除批次维度
        inputs = {key: inputs[key].squeeze(0) for key in inputs}
        
        return inputs

# 创建数据集实例
train_dataset = SpeechDataset(train_files, data_dir)
val_dataset = SpeechDataset(val_files, data_dir)
processor = train_dataset.processor  # 提取processor供后续使用

# 加载预训练模型
model = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

# 设置模型为训练模式
model.train()
model.to(device)

# 定义数据整理函数
def data_collator(batch):
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # 找出批次中最长的音频长度
    max_length = max(item['input_values'].shape[0] for item in batch)
    
    # 找出批次中最长的标签长度
    max_label_length = max(item['labels'].shape[0] for item in batch)
    
    # 初始化填充后的张量
    padded_input_values = []
    padded_attention_mask = []
    padded_labels = []
    
    for item in batch:
        # 填充音频输入
        input_length = item['input_values'].shape[0]
        padding_length = max_length - input_length
        
        if padding_length > 0:
            padded_input = torch.cat([
                item['input_values'],
                torch.zeros(padding_length, dtype=item['input_values'].dtype)
            ])
            padded_attention = torch.cat([
                item['attention_mask'],
                torch.zeros(padding_length, dtype=item['attention_mask'].dtype)
            ])
        else:
            padded_input = item['input_values']
            padded_attention = item['attention_mask']
        
        # 填充标签
        label_length = item['labels'].shape[0]
        label_padding_length = max_label_length - label_length
        
        if label_padding_length > 0:
            padded_label = torch.cat([
                item['labels'],
                torch.full((label_padding_length,), -100, dtype=item['labels'].dtype)
            ])
        else:
            padded_label = item['labels']
        
        padded_input_values.append(padded_input)
        padded_attention_mask.append(padded_attention)
        padded_labels.append(padded_label)
    
    return {
        'input_values': torch.stack(padded_input_values),
        'attention_mask': torch.stack(padded_attention_mask),
        'labels': torch.stack(padded_labels)
    }

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 学习率调度器
num_epochs = 8  # 适当增加训练轮数以获得更好效果
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

# 早停机制
early_stopping_patience = 3
best_val_wer = float('inf')  # 改为跟踪WER作为早停指标
patience_counter = 0

# 用于记录训练历史的列表
train_history = []

# 定义验证集评估函数（计算Loss和WER）
def evaluate_on_validation_set(model, dataloader, processor, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on validation set"):
            if batch is None:
                continue
                
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 计算损失
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
            
            # 生成预测
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # 解码预测结果和标签（转换为文本）
            predictions = processor.batch_decode(predicted_ids)
            references = processor.batch_decode(labels, group_tokens=False)  # 不解码成组，保持原始标签
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # 计算平均损失和WER
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    
    return avg_loss, wer, all_predictions, all_references

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    
    for batch in train_bar:
        if batch is None:
            continue
            
        # 将数据移动到设备
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播 - 使用混合精度
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        # 反向传播 - 使用混合精度
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        
        train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
    
    # 验证阶段（计算Loss和WER）
    avg_val_loss, val_wer, _, _ = evaluate_on_validation_set(model, val_loader, processor, device)
    
    # 记录训练历史
    train_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_wer': val_wer,  # 记录WER
        'learning_rate': lr_scheduler.get_last_lr()[0] if len(train_loader) > 0 else 0
    })
    
    # 打印当前epoch的评估结果
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val WER: {val_wer:.4f} (越低越好，0表示完全正确)")
    
    # 早停检查（基于WER）
    if val_wer < best_val_wer:
        best_val_wer = val_wer
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        print(f"保存最佳模型 (WER: {best_val_wer:.4f})")
    else:
        patience_counter += 1
        print(f"早停计数器: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break

# 保存训练历史到Excel（包含WER）
history_df = pd.DataFrame(train_history)
history_df.to_excel("/kaggle/working/training_history.xlsx", index=False)
print("\n训练历史已保存到 /kaggle/working/training_history.xlsx")

# 加载最佳模型并在验证集上做最终评估
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
final_val_loss, final_val_wer, all_preds, all_refs = evaluate_on_validation_set(model, val_loader, processor, device)

print(f"\n最终模型在验证集上的表现:")
print(f"最终验证Loss: {final_val_loss:.4f}")
print(f"最终验证WER: {final_val_wer:.4f} (错误率：{final_val_wer*100:.2f}%)")

# 随机选择10个验证样本进行详细测试
val_file_list = [val_files.dataset[i] for i in val_files.indices]
test_samples = random.sample(val_file_list, min(10, len(val_file_list)))

# 创建测试结果列表
test_results = []

# 进行详细测试
for i, wav_file in enumerate(test_samples):
    wav_path = os.path.join(data_dir, wav_file)
    trn_path = os.path.join(data_dir, wav_file + ".trn")
    
    # 读取真实文本
    try:
        with open(trn_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            true_text = lines[0].strip() if lines else ""
    except Exception as e:
        print(f"读取标签文件错误 {trn_path}: {e}")
        true_text = ""
    
    # 读取和处理音频
    try:
        speech_array, sampling_rate = sf.read(wav_path)
        speech_array = speech_array.astype(np.float32)
    except Exception as e:
        print(f"读取音频文件错误 {wav_path}: {e}")
        continue
    
    if len(speech_array.shape) > 1:
        speech_array = np.mean(speech_array, axis=1)
    
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
    
    # 模型预测
    inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(inputs.input_values.to(device)).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(predicted_ids)[0]
    
    # 保存结果
    test_results.append({
        'audio_file': wav_file,
        'true_text': true_text,
        'predicted_text': prediction,
        'audio_array': speech_array,
        'sampling_rate': 16000
    })
    
    print(f"\nSample {i+1}:")
    print(f"True: {true_text}")
    print(f"Predicted: {prediction}")
    print("-" * 50)

# 保存测试结果到Excel
test_results_df = pd.DataFrame([{
    'audio_file': r['audio_file'],
    'true_text': r['true_text'],
    'predicted_text': r['predicted_text']
} for r in test_results])
test_results_df.to_excel("/kaggle/working/test_results.xlsx", index=False)
print("\n测试结果已保存到 /kaggle/working/test_results.xlsx")

# 显示测试样本的音频和文本
for i, result in enumerate(test_results):
    print(f"\nSample {i+1}: {result['audio_file']}")
    print(f"True text: {result['true_text']}")
    print(f"Predicted text: {result['predicted_text']}")
    
    # 显示音频播放器
    try:
        display(Audio(result['audio_array'], rate=result['sampling_rate']))
    except:
        print("当前环境不支持音频显示")
    print("=" * 80)