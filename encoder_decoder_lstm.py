import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 设置matplotlib支持中文显示
# 根据CSDN博客 https://blog.csdn.net/weixin_46474921/article/details/123783987 的解决方案
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统推荐字体
plt.rcParams['axes.unicode_minus'] = False  # 解决中文字体下坐标轴负数的负号显示问题

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

    # 8. Embedding质量分析与可视化
    print("\n" + "=" * 50)
    print("Embedding质量分析与可视化...")
    
    # 从模型中获取源语言的embedding矩阵和词汇表
    src_embedding_matrix = model.encoder.embedding.weight.data.cpu()
    
    # 先进行质量分析
    analysis_results = analyze_embedding_quality(src_embedding_matrix, src_vocab, top_k=8)
    
    # 再进行可视化
    print(f"\n{'='*60}")
    print("🎨 生成t-SNE可视化图...")
    print(f"{'='*60}")
    visualize_embeddings(src_embedding_matrix, src_vocab, method='tsne', title="Source Language Embedding Visualization (t-SNE)")

def analyze_embedding_quality(embedding_matrix, vocab, top_k=5):
    """分析embedding质量和聚类效果"""
    print(f"\n{'='*60}")
    print("📊 Embedding质量分析")
    print(f"{'='*60}")
    
    # 计算所有词汇的余弦相似度矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 过滤掉特殊标记，只分析实际词汇
    real_words = []
    real_indices = []
    real_embeddings = []
    
    for idx, word in vocab.idx2word.items():
        if idx < embedding_matrix.shape[0] and word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
            real_words.append(word)
            real_indices.append(idx)
            real_embeddings.append(embedding_matrix[idx].numpy())
    
    if len(real_embeddings) < 2:
        print("⚠️  实际词汇数量不足，无法进行聚类分析")
        return
    
    real_embeddings = np.array(real_embeddings)
    similarity_matrix = cosine_similarity(real_embeddings)
    
    print(f"\n🔍 词汇相似度分析 (共{len(real_words)}个词汇)")
    print("-" * 50)
    
    # 1. 找出最相似的词对
    most_similar_pairs = []
    for i in range(len(real_words)):
        for j in range(i+1, len(real_words)):
            similarity = similarity_matrix[i][j]
            most_similar_pairs.append((real_words[i], real_words[j], similarity))
    
    # 按相似度排序
    most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n📈 最相似的{min(top_k, len(most_similar_pairs))}对词汇:")
    for i, (word1, word2, sim) in enumerate(most_similar_pairs[:top_k]):
        print(f"   {i+1}. '{word1}' ↔ '{word2}': {sim:.4f}")
    
    # 2. 分析特定主题词汇的聚集度
    print(f"\n🎯 主题词汇聚集分析:")
    print("-" * 30)
    
    # 定义主题词汇组
    theme_groups = {
        "时间天气": ["今天", "天气", "好"],
        "AI技术": ["人工", "智能", "机器", "学习"],
        "NLP": ["自然", "语言", "处理"],
        "情感": ["爱", "有趣", "强大"]
    }
    
    for theme_name, words in theme_groups.items():
        # 找出该主题中存在的词汇
        existing_words = [w for w in words if w in real_words]
        if len(existing_words) >= 2:
            # 计算组内平均相似度
            indices = [real_words.index(w) for w in existing_words]
            group_similarities = []
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    group_similarities.append(similarity_matrix[indices[i]][indices[j]])
            
            avg_similarity = np.mean(group_similarities)
            print(f"   {theme_name}: {existing_words} → 平均相似度: {avg_similarity:.4f}")
    
    # 3. 检查共现词汇的相似度
    print(f"\n🔗 训练数据共现词汇相似度:")
    print("-" * 35)
    
    cooccurrence_pairs = [
        ("我", "爱"), ("今天", "天气"), ("天气", "好"),
        ("机器", "学习"), ("深度", "学习"), ("人工", "智能")
    ]
    
    for word1, word2 in cooccurrence_pairs:
        if word1 in real_words and word2 in real_words:
            idx1, idx2 = real_words.index(word1), real_words.index(word2)
            similarity = similarity_matrix[idx1][idx2]
            print(f"   '{word1}' ↔ '{word2}': {similarity:.4f}")
    
    # 4. 统计分析
    print(f"\n📊 Embedding统计特性:")
    print("-" * 25)
    
    # 向量范数分析
    norms = np.linalg.norm(real_embeddings, axis=1)
    print(f"   向量范数 - 均值: {np.mean(norms):.4f}, 标准差: {np.std(norms):.4f}")
    
    # 整体相似度分布
    upper_triangle = similarity_matrix[np.triu_indices(len(real_words), k=1)]
    print(f"   相似度分布 - 均值: {np.mean(upper_triangle):.4f}, 标准差: {np.std(upper_triangle):.4f}")
    print(f"   相似度范围: [{np.min(upper_triangle):.4f}, {np.max(upper_triangle):.4f}]")
    
    # 5. 异常检测
    print(f"\n⚠️  异常向量检测:")
    print("-" * 20)
    
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    outlier_threshold = 2.0  # 2倍标准差
    
    outliers = []
    for i, (word, norm) in enumerate(zip(real_words, norms)):
        if abs(norm - mean_norm) > outlier_threshold * std_norm:
            outliers.append((word, norm))
    
    if outliers:
        print(f"   发现{len(outliers)}个异常向量:")
        for word, norm in outliers:
            print(f"     '{word}': 范数 = {norm:.4f}")
    else:
        print("   ✅ 未发现明显异常向量")

    return {
        'similarity_matrix': similarity_matrix,
        'most_similar_pairs': most_similar_pairs[:top_k],
        'real_words': real_words,
        'statistics': {
            'mean_norm': np.mean(norms),
            'mean_similarity': np.mean(upper_triangle),
            'std_similarity': np.std(upper_triangle)
        }
    }

def visualize_embeddings(embedding_matrix, vocab, method='tsne', title='Embedding Visualization', num_words_to_annotate=20):
    """使用t-SNE或PCA可视化embedding"""
    # 确保词汇表中的词少于或等于实际embedding矩阵中的行数
    # 通常，vocab.vocab_size 会是 embedding_matrix.shape[0]
    # 但为了安全，我们取两者中较小的值，并排除特殊标记（如果它们影响可视化）
    
    # 获取所有词（排除特殊标记如<PAD>, <SOS>, <EOS>, <UNK>，如果它们在词汇表索引的开头）
    # 我们假设特殊标记的索引较小，如果 visualize_embeddings 只关注非特殊词汇
    
    # 过滤掉权重全为零的向量 (通常是 padding_idx)
    # 同时收集有效的词和它们的索引
    valid_indices = []
    valid_words = []
    
    # 创建一个列表来保存所有词的向量
    all_vectors_list = []
    
    # 获取所有词的列表，按索引顺序
    # vocab.idx2word 是一个字典，我们需要按索引排序的词
    # 我们只取词汇表中实际存在的词汇，直到 embedding_matrix.shape[0]
    # 通常 vocab.vocab_size 应该等于 embedding_matrix.shape[0]
    
    words_to_process_indices = sorted([idx for idx in vocab.idx2word.keys() if idx < embedding_matrix.shape[0]])

    for idx in words_to_process_indices:
        word = vocab.idx2word[idx]
        vector = embedding_matrix[idx]
        # 排除<PAD>等特殊token的可视化, 通常 padding_idx 为 0
        if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] or torch.any(vector != 0):
             # 只有当词不是特殊词，或者向量不全为0时才添加
            # 实际上，对于非特殊词，向量不应全为0，除非embed_size很小或特殊情况
            # 而对于<PAD>，我们通常希望排除它，除非特别想观察它
            if not (word == '<PAD>' and torch.all(vector == 0)):
                 all_vectors_list.append(vector.numpy()) # TSNE/PCA需要numpy数组
                 valid_words.append(word)
                 valid_indices.append(idx) # 虽然未使用，但保留以备将来之需

    if not all_vectors_list:
        print("没有有效的词向量可供可视化。")
        return

    embeddings_to_visualize = np.array(all_vectors_list)

    if embeddings_to_visualize.shape[0] < 2:
        print(f"有效的词向量数量 ({embeddings_to_visualize.shape[0]}) 不足以进行降维可视化。")
        return

    # 降维到2D
    # t-SNE对于少于 perplexity+1 个样本会出问题，通常perplexity在5-50之间
    # PCA没有这个限制
    n_samples = embeddings_to_visualize.shape[0]
    
    if method == 'tsne':
        # 对于非常小的样本量，TSNE可能失败或产生无意义的结果
        # 调整 perplexity, n_iter, learning_rate
        perplexity_value = min(30.0, float(n_samples - 1)) # Perplexity must be less than n_samples
        if perplexity_value <= 0: # 如果只有一个点或没有点
             print(f"样本数量 ({n_samples}) 过少，无法使用t-SNE。")
             if n_samples > 1 and embeddings_to_visualize.ndim == 2 and embeddings_to_visualize.shape[1] >=2:
                 print("尝试使用PCA替代...")
                 method = 'pca' # 尝试PCA
             else:
                 return # 确实无法可视化
        
        if method == 'tsne': # 再次检查，因为可能在上面被改为pca
            try:
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, 
                               max_iter=300, learning_rate=200) # 修复sklearn参数：n_iter改为max_iter
                embeddings_2d = reducer.fit_transform(embeddings_to_visualize)
            except Exception as e:
                print(f"t-SNE执行失败: {e}。尝试使用PCA。")
                if n_samples > 1 and embeddings_to_visualize.ndim == 2 and embeddings_to_visualize.shape[1] >=2: # 确保PCA可以运行
                    method = 'pca'
                else:
                    return


    if method == 'pca': # 如果原始方法是PCA，或者t-SNE失败后转为PCA
        if n_samples < 2 or embeddings_to_visualize.shape[1] < 2: # PCA至少需要2个样本和2个特征
            print("样本或特征数量不足以进行PCA可视化。")
            return
        try:
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings_to_visualize)
        except Exception as e:
            print(f"PCA执行失败: {e}")
            return


    if embeddings_2d is None or embeddings_2d.shape[0] == 0:
        print("降维失败，没有生成2D嵌入。")
        return

    # 绘制散点图
    plt.figure(figsize=(12, 10)) # 增大图像尺寸以容纳更多标签
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=30) # 增大点的大小
    
    # 添加词汇标签 (只注释一部分词以避免拥挤)
    words_to_annotate_actual = min(num_words_to_annotate, len(valid_words))
    
    # 为了更好的可读性，选择一些词进行标注，例如均匀间隔的或者随机选择的
    # 这里我们简单选择前 num_words_to_annotate_actual 个词
    indices_to_annotate = np.random.choice(len(valid_words), size=words_to_annotate_actual, replace=False)


    for i in indices_to_annotate:
        plt.annotate(valid_words[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout() # 调整布局以防止标签溢出
    # 保存图像而不是显示，以便在无GUI环境运行
    try:
        plt.savefig("embedding_visualization.png")
        print(f"Embedding可视化图像已保存到 embedding_visualization.png")
    except Exception as e:
        print(f"保存图像失败: {e}")
    # plt.show() # 在脚本中通常不直接调用show()，除非是交互式运行

if __name__ == "__main__":
    main()
