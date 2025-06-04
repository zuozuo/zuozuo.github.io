import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

class Vocabulary:
    """词汇表类，用于文本和数字之间的转换"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def sentence_to_indices(self, sentence):
        return [self.word2idx.get(word, self.word2idx['<UNK>']) 
                for word in sentence.split()]
    
    def indices_to_sentence(self, indices):
        return ' '.join([self.idx2word[idx] for idx in indices 
                        if idx not in [0, 1, 2]])  # 排除特殊标记

class LSTMEncoder(nn.Module):
    """LSTM编码器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # LSTM层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=False)
        
    def forward(self, input_seq, input_lengths):
        # input_seq: [batch_size, seq_len]
        batch_size = input_seq.size(0)
        
        # 词嵌入
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, embed_size]
        
        # 打包序列以处理不同长度的输入
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM前向传播
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 返回最后的隐状态作为固定长度的向量表示
        # hidden: [num_layers, batch_size, hidden_size]
        # 我们取最后一层的隐状态
        context_vector = hidden[-1]  # [batch_size, hidden_size]
        
        return context_vector, (hidden, cell)

class LSTMDecoder(nn.Module):
    """LSTM解码器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, encoder_hidden, target_seq=None, max_length=20):
        batch_size = encoder_hidden[0].size(1)
        
        if target_seq is not None:
            # 训练模式：使用目标序列
            embedded = self.embedding(target_seq)
            output, _ = self.lstm(embedded, encoder_hidden)
            output = self.output_projection(output)
            return output
        else:
            # 推理模式：逐步生成
            outputs = []
            hidden = encoder_hidden
            input_token = torch.ones(batch_size, 1, dtype=torch.long) * 1  # <SOS>
            
            for _ in range(max_length):
                embedded = self.embedding(input_token)
                output, hidden = self.lstm(embedded, hidden)
                output = self.output_projection(output)
                
                # 取概率最大的词作为下一个输入
                input_token = output.argmax(dim=-1)
                outputs.append(output)
                
                # 如果生成了<EOS>，提前停止
                if (input_token == 2).all():
                    break
            
            return torch.cat(outputs, dim=1)

class Seq2SeqModel(nn.Module):
    """序列到序列模型"""
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_seq, input_lengths, target_seq=None, max_length=20):
        # 编码阶段：将输入序列压缩成固定长度向量
        context_vector, encoder_hidden = self.encoder(input_seq, input_lengths)
        
        print(f"输入序列形状: {input_seq.shape}")
        print(f"上下文向量形状: {context_vector.shape}")
        print(f"编码器隐状态形状: {encoder_hidden[0].shape}")
        
        # 解码阶段：基于上下文向量生成输出序列
        output = self.decoder(encoder_hidden, target_seq, max_length)
        
        return output, context_vector

# 创建示例数据
def create_sample_data():
    """创建示例训练数据"""
    # 中文到英文的翻译示例
    data_pairs = [
        ("我 爱 自然 语言 处理", "I love natural language processing"),
        ("今天 天气 很 好", "Today weather is good"),
        ("机器 学习 很 有趣", "Machine learning is interesting"),
        ("深度 学习 很 强大", "Deep learning is powerful"),
        ("人工 智能 改变 世界", "AI changes the world"),
    ]
    return data_pairs

class TranslationDataset(Dataset):
    """翻译数据集"""
    def __init__(self, data_pairs, src_vocab, tgt_vocab):
        self.data_pairs = data_pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.data_pairs[idx]
        
        src_indices = self.src_vocab.sentence_to_indices(src_sentence)
        tgt_indices = [1] + self.tgt_vocab.sentence_to_indices(tgt_sentence) + [2]  # 添加<SOS>和<EOS>
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    """数据批处理函数"""
    src_batch, tgt_batch = zip(*batch)
    
    # 计算序列长度
    src_lengths = [len(seq) for seq in src_batch]
    tgt_lengths = [len(seq) for seq in tgt_batch]
    
    # 填充序列
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, torch.tensor(src_lengths), tgt_batch, torch.tensor(tgt_lengths)

def main():
    # 1. 准备数据
    print("=" * 50)
    print("准备数据...")
    data_pairs = create_sample_data()
    
    # 构建词汇表
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    for src, tgt in data_pairs:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
    
    print(f"源语言词汇表大小: {src_vocab.vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab.vocab_size}")
    
    # 2. 创建数据集和数据加载器
    dataset = TranslationDataset(data_pairs, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # 3. 定义模型参数
    print("\n" + "=" * 50)
    print("初始化模型...")
    embed_size = 64
    hidden_size = 128
    num_layers = 1
    
    # 4. 创建编码器和解码器
    encoder = LSTMEncoder(src_vocab.vocab_size, embed_size, hidden_size, num_layers)
    decoder = LSTMDecoder(tgt_vocab.vocab_size, embed_size, hidden_size, num_layers)
    model = Seq2SeqModel(encoder, decoder)
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. 训练模型
    print("\n" + "=" * 50)
    print("开始训练...")
    model.train()
    
    for epoch in range(50):
        total_loss = 0
        for batch_idx, (src_batch, src_lengths, tgt_batch, tgt_lengths) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 准备目标序列（用于teacher forcing）
            decoder_input = tgt_batch[:, :-1]  # 去掉最后一个token
            decoder_target = tgt_batch[:, 1:]  # 去掉第一个token(<SOS>)
            
            # 前向传播
            output, context_vector = model(src_batch, src_lengths, decoder_input)
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), 
                           decoder_target.reshape(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader):.4f}')
    
    # 7. 测试模型
    print("\n" + "=" * 50)
    print("测试模型...")
    model.eval()
    
    test_sentences = ["我 爱 自然 语言 处理", "今天 天气 很 好"]
    
    with torch.no_grad():
        for test_sentence in test_sentences:
            print(f"\n输入: {test_sentence}")
            
            # 将句子转换为索引
            src_indices = src_vocab.sentence_to_indices(test_sentence)
            src_tensor = torch.tensor(src_indices).unsqueeze(0)  # 添加batch维度
            src_length = torch.tensor([len(src_indices)])
            
            # 生成翻译
            output, context_vector = model(src_tensor, src_length, max_length=10)
            
            # 将输出转换为单词
            predicted_indices = output.argmax(dim=-1).squeeze(0).tolist()
            predicted_sentence = tgt_vocab.indices_to_sentence(predicted_indices)
            
            print(f"输出: {predicted_sentence}")
            print(f"上下文向量维度: {context_vector.shape}")
            print(f"上下文向量值: {context_vector.squeeze().numpy()[:5]}...")  # 只显示前5个值

if __name__ == "__main__":
    main()
