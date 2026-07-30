"""
edrl.py  ——  实体描述表征学习（EDRL）模块
==============================================
严格按照博士论文第二章 §2.2.2 实现，包含：

  阶段一：继续预训练（Contrastive Pre-training）
    - 输入：头尾实体描述拼接形成锚点 A；所有候选关系描述
    - 正样本 R+：该实体对真实存在的关系
    - 负样本 Rs-：随机采样 n_neg 个不成立的关系（论文默认 10 个）
    - 损失：InfoNCE / 对比学习损失（公式 2-6）
      L_pt = -log [ exp(sim(A,R+)/τ) / (exp(sim(A,R+)/τ) + Σ exp(sim(A,Rs-)/τ)) ]

  阶段二：关系分类微调（Fine-tuning）
    - 输入：[CLS] D_h [SEP] D_t（公式 2-7）
    - 提取 [CLS] 隐状态 C_i
    - 线性分类层 W：S_τ = Softmax(C_i W^T)（公式 2-8）
    - 损失：交叉熵（公式 2-9）

  特征提取（Feature Extraction）
    - 固定微调后的 BERT*，对每个实体/关系单独编码
    - 通过 Linear_F 降维到 KG 嵌入空间（公式 2-10）
    - 输出：entity_features_pretrained.npy  /  relation_features_pretrained.npy

使用方法：
    python edrl.py --dataset wn18 --bert_model /path/to/bert-base-uncased
    python edrl.py --dataset FB15k-237 --bert_model /path/to/bert-base-uncased
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# ─────────────────────────────────────────────────────────────────────────────
# 数据集配置
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CONFIG = {
    'wn18': {
        'data_dir':         'data/wn18',
        'entity2text':      'data_with_text/WN18/entity2text.txt',
        'relation2text':    'data_with_text/WN18/relation2text.txt',
        'text_format':      'tab_id_text',
    },
    'wn18rr': {
        'data_dir':         'data/wn18rr',
        'entity2text':      'data_with_text/WN18RR/entity2text.txt',
        'relation2text':    'data_with_text/WN18RR/relation2text.txt',
        'text_format':      'tab_id_text',
    },
    'FB15k-237': {
        'data_dir':         'data/FB15k-237',
        'entity2text':      'data_with_text/FB15k-237/entity2text.txt',
        'relation2text':    'data_with_text/FB15k-237/relation2text.txt',
        'text_format':      'tab_id_text',
    },
    'NELL995': {
        'data_dir':         'data/NELL995',
        'entity2text':      None,
        'relation2text':    None,
        'text_format':      'name_only',
    },
    'LegalPP': {
        'data_dir':         'data/LegalPP',
        'entity2text':      'data_with_text/LegalPP/entity2text.txt',
        'relation2text':    'data_with_text/LegalPP/relation2text.txt',
        'text_format':      'tab_name_text',
    },
    'LegalPP_link': {
        'data_dir':         'data/LegalPP_link',
        'entity2text':      'data_with_text/LegalPP_link/entity2text.txt',
        'relation2text':    'data_with_text/LegalPP_link/relation2text.txt',
        'text_format':      'tab_name_text',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载工具
# ─────────────────────────────────────────────────────────────────────────────
def load_dict(path):
    """读取 {id}\t{name} 格式的字典文件，返回 id->name 列表（按 id 排序）"""
    id2name = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split('\t', 1)
            id2name[int(idx)] = name
    return [id2name[i] for i in range(len(id2name))]


def load_text_map(path, fmt):
    """读取文本描述文件，返回 {name: text} 字典"""
    name2text = {}
    if path is None or not os.path.exists(path):
        return name2text
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if fmt in ('tab_id_text', 'tab_name_text'):
                # tab_id_text:   id\ttext  (WN18/FB15k-237)
                # tab_name_text: name\ttext (LegalPP)
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    name2text[parts[0]] = parts[1]
    return name2text


def load_triplets(path):
    """读取三元组文件，返回 [(h_name, r_name, t_name), ...]"""
    triplets = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triplets.append((parts[0], parts[1], parts[2]))
    return triplets


def get_text(name, text_map, fmt):
    """获取实体/关系的文本描述，无描述时使用名称本身"""
    if name in text_map:
        return text_map[name]
    return name.replace('_', ' ').replace('/', ' ').strip()


# ─────────────────────────────────────────────────────────────────────────────
# BM25 正样本选取（论文 §2.3.2：k1=0.9, b=0.4）
# ─────────────────────────────────────────────────────────────────────────────
def build_bm25_index(relation_names, relation_texts, k1=0.9, b=0.4):
    """
    对所有关系描述建立 BM25 索引。
    论文参数：词频平滑 k1=0.9，文档长度归一化 b=0.4。
    返回 (bm25_obj, tokenized_corpus)。
    """
    if not HAS_BM25:
        raise ImportError('rank_bm25 not installed. Run: pip install rank_bm25')
    corpus = []
    for rn in relation_names:
        text = get_text(rn, relation_texts, None)
        tokens = text.lower().split()
        corpus.append(tokens)
    bm25 = BM25Okapi(corpus, k1=k1, b=b)
    return bm25, corpus


def bm25_top1_relation(anchor_text, bm25, relation_names, true_r_name):
    """
    用 BM25 对锚点文本检索候选关系，返回排名最高的正确关系名称。
    若 BM25 最高分关系恰好是真实关系，则直接返回；
    否则仍返回真实关系（保证正样本正确性，符合论文"选取 1 条正确关系"的描述）。
    """
    query_tokens = anchor_text.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked_idx = np.argsort(scores)[::-1]
    # 取排名最高的真实关系（论文：从排序结果中选取 1 条排名最高的正确关系）
    for idx in ranked_idx:
        if relation_names[idx] == true_r_name:
            return relation_names[idx]
    return true_r_name  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# 阶段一：对比学习预训练 Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveDataset(Dataset):
    """
    每个样本由一个三元组 (h, r, t) 构成：
      - 锚点 A = concat(D_h, D_t)（头尾实体描述拼接）
      - 正样本 R+：BM25 检索排名最高的正确关系描述（论文 §2.3.2）
      - 负样本 Rs-：随机采样 n_neg 个不成立关系的描述（论文默认 10 个）
    """
    def __init__(self, triplets, entity_texts, relation_texts,
                 relation_names, n_neg=10, bm25=None):
        self.triplets = triplets
        self.entity_texts = entity_texts
        self.relation_texts = relation_texts
        self.relation_names = relation_names
        self.n_neg = n_neg
        self.n_relations = len(relation_names)
        self.bm25 = bm25  # BM25 索引，None 时退化为直接使用真实关系

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        h_name, r_name, t_name = self.triplets[idx]

        # 锚点：头尾实体描述拼接（用空格分隔）
        h_text = get_text(h_name, self.entity_texts, None)
        t_text = get_text(t_name, self.entity_texts, None)
        anchor_text = h_text + ' ' + t_text

        # 正样本：BM25 检索排名最高的正确关系（论文 §2.3.2）
        if self.bm25 is not None:
            pos_r_name = bm25_top1_relation(anchor_text, self.bm25, self.relation_names, r_name)
        else:
            pos_r_name = r_name
        pos_text = get_text(pos_r_name, self.relation_texts, None)

        # 负样本：随机采样 n_neg 个不等于 r_name 的关系
        neg_names = [rn for rn in self.relation_names if rn != r_name]
        neg_sampled = random.sample(neg_names, min(self.n_neg, len(neg_names)))
        neg_texts = [get_text(rn, self.relation_texts, None) for rn in neg_sampled]

        return anchor_text, pos_text, neg_texts


def contrastive_collate_fn(batch, tokenizer, max_length=64):
    """将 ContrastiveDataset 的 batch 编码为 BERT 输入"""
    anchors, pos_list, neg_list_of_lists = zip(*batch)
    n_neg = len(neg_list_of_lists[0])

    # 编码锚点
    anchor_enc = tokenizer(list(anchors), padding=True, truncation=True,
                           max_length=max_length, return_tensors='pt')
    # 编码正样本
    pos_enc = tokenizer(list(pos_list), padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
    # 编码负样本（展平后编码，再 reshape）
    neg_flat = [text for neg_texts in neg_list_of_lists for text in neg_texts]
    neg_enc = tokenizer(neg_flat, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')

    return anchor_enc, pos_enc, neg_enc, n_neg


# ─────────────────────────────────────────────────────────────────────────────
# 阶段二：关系分类微调 Dataset
# ─────────────────────────────────────────────────────────────────────────────
class FineTuneDataset(Dataset):
    """
    每个样本由一个三元组 (h, r, t) 构成：
      - 输入：[CLS] D_h [SEP] D_t（公式 2-7）
      - 标签：关系 r 对应的整数 id
    """
    def __init__(self, triplets, entity_texts, relation_name2id):
        self.triplets = triplets
        self.entity_texts = entity_texts
        self.relation_name2id = relation_name2id

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        h_name, r_name, t_name = self.triplets[idx]
        h_text = get_text(h_name, self.entity_texts, None)
        t_text = get_text(t_name, self.entity_texts, None)
        label = self.relation_name2id[r_name]
        return h_text, t_text, label


def finetune_collate_fn(batch, tokenizer, max_length=128):
    """将 FineTuneDataset 的 batch 编码为 BERT 输入"""
    h_texts, t_texts, labels = zip(*batch)
    # 按照论文公式 (2-7)：[CLS] D_h [SEP] D_t
    enc = tokenizer(list(h_texts), list(t_texts),
                    padding=True, truncation=True,
                    max_length=max_length, return_tensors='pt')
    labels = torch.LongTensor(labels)
    return enc, labels


# ─────────────────────────────────────────────────────────────────────────────
# EDRL 模型
# ─────────────────────────────────────────────────────────────────────────────
class EDRLModel(nn.Module):
    """
    EDRL 模型：BERT + 线性分类头（微调阶段）
    预训练阶段直接使用 BERT 的 [CLS] 输出计算余弦相似度
    """
    def __init__(self, bert_model_path, n_relations, kg_dim=64):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        hidden_size = self.bert.config.hidden_size  # 768

        # 微调阶段的线性分类层 W（公式 2-8）
        self.classifier = nn.Linear(hidden_size, n_relations)
        nn.init.xavier_uniform_(self.classifier.weight)

        # Linear_F：降维到 KG 嵌入空间（公式 2-10）
        self.linear_f = nn.Linear(hidden_size, kg_dim)
        nn.init.xavier_uniform_(self.linear_f.weight)

    def encode(self, enc_inputs):
        """提取 [CLS] 隐状态"""
        outputs = self.bert(**enc_inputs)
        return outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

    def forward_contrastive(self, anchor_enc, pos_enc, neg_enc, n_neg, temperature=0.07):
        """
        对比学习前向传播（公式 2-6）
        L_pt = -log [ exp(sim(A,R+)/τ) / (exp(sim(A,R+)/τ) + Σ exp(sim(A,Rs-)/τ)) ]
        """
        batch_size = anchor_enc['input_ids'].shape[0]

        # 编码锚点 A、正样本 R+、负样本 Rs-
        a_vec = self.encode(anchor_enc)                          # [B, H]
        r_pos_vec = self.encode(pos_enc)                         # [B, H]
        r_neg_flat = self.encode(neg_enc)                        # [B*n_neg, H]
        r_neg_vec = r_neg_flat.view(batch_size, n_neg, -1)       # [B, n_neg, H]

        # 余弦相似度
        a_norm = F.normalize(a_vec, dim=-1)                      # [B, H]
        pos_norm = F.normalize(r_pos_vec, dim=-1)                # [B, H]
        neg_norm = F.normalize(r_neg_vec, dim=-1)                # [B, n_neg, H]

        sim_pos = torch.sum(a_norm * pos_norm, dim=-1) / temperature        # [B]
        sim_neg = torch.bmm(neg_norm, a_norm.unsqueeze(-1)).squeeze(-1) / temperature  # [B, n_neg]

        # InfoNCE Loss
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # [B, 1+n_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward_finetune(self, enc_inputs, labels):
        """
        关系分类微调前向传播（公式 2-7, 2-8, 2-9）
        C_i = BERT([CLS] D_h [SEP] D_t)[CLS]
        S_τ = Softmax(C_i W^T)
        L_ft = CrossEntropy
        """
        cls_vec = self.encode(enc_inputs)                        # [B, H]
        logits = self.classifier(cls_vec)                        # [B, n_relations]
        loss = F.cross_entropy(logits, labels)
        return loss, logits


# ─────────────────────────────────────────────────────────────────────────────
# 特征提取（公式 2-10）
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, tokenizer, texts, device, batch_size=64, max_length=64):
    """
    对每个文本单独编码，提取 [CLS] 向量，再通过 Linear_F 降维。
    对应论文公式 (2-10)：E = Linear_F(BERT*(text))
    """
    model.eval()
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        cls_vec = model.encode(enc)                              # [b, H]
        feat = model.linear_f(cls_vec)                           # [b, kg_dim]
        all_feats.append(feat.cpu().float().numpy())
        if (i // batch_size) % 10 == 0:
            print(f'  extracted {min(i + batch_size, len(texts))}/{len(texts)}')
    return np.concatenate(all_feats, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='EDRL: Entity Description Representation Learning')
    parser.add_argument('--dataset',        type=str,   required=True,
                        help='Dataset name: wn18 / wn18rr / FB15k-237 / NELL995 / LegalPP / LegalPP_link')
    parser.add_argument('--bert_model',     type=str,   default='bert-base-uncased',
                        help='Path to pretrained BERT model')
    parser.add_argument('--kg_dim',         type=int,   default=400,
                        help='KG embedding dimension (Linear_F output dim, 论文默认 400)')
    parser.add_argument('--pt_epochs',      type=int,   default=5,
                        help='Contrastive pre-training epochs')
    parser.add_argument('--ft_epochs',      type=int,   default=5,
                        help='Fine-tuning epochs')
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--lr_pt',          type=float, default=2e-5,
                        help='Learning rate for pre-training')
    parser.add_argument('--lr_ft',          type=float, default=2e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--temperature',    type=float, default=0.05,
                        help='Temperature τ for contrastive loss (论文§2.3.2 参照 SimCSE, 默认 0.05)')
    parser.add_argument('--n_neg',          type=int,   default=10,
                        help='Number of negative relation samples (论文默认 10)')
    parser.add_argument('--bm25_k1',        type=float, default=0.9,
                        help='BM25 term frequency smoothing k1 (论文§2.3.2, 默认 0.9)')
    parser.add_argument('--bm25_b',         type=float, default=0.4,
                        help='BM25 document length normalization b (论文§2.3.2, 默认 0.4)')
    parser.add_argument('--max_length_pt',  type=int,   default=64,
                        help='Max token length for pre-training')
    parser.add_argument('--max_length_ft',  type=int,   default=128,
                        help='Max token length for fine-tuning ([CLS] D_h [SEP] D_t)')
    parser.add_argument('--max_length_ext', type=int,   default=64,
                        help='Max token length for feature extraction')
    parser.add_argument('--no_pretrain',    action='store_true',
                        help='Skip contrastive pre-training, go directly to fine-tuning')
    parser.add_argument('--num_workers',    type=int,   default=4,
                        help='DataLoader num_workers for parallel data loading')
    parser.add_argument('--fp16',           action='store_true',
                        help='Use mixed precision (AMP) training for faster GPU computation')
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = args.dataset
    if dataset not in DATASET_CONFIG:
        raise ValueError(f'Unknown dataset: {dataset}')

    cfg = DATASET_CONFIG[dataset]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, cfg['data_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[Dataset] {dataset}  |  [Device] {device}')

    # ── 加载字典 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading entity/relation dicts ...')
    entity_names = load_dict(os.path.join(data_dir, 'entities.dict'))
    relation_names = load_dict(os.path.join(data_dir, 'relations.dict'))
    relation_name2id = {name: idx for idx, name in enumerate(relation_names)}
    n_relations = len(relation_names)
    print(f'  Entities: {len(entity_names)}  |  Relations: {n_relations}')

    # ── 加载文本描述 ──────────────────────────────────────────────────────────
    print('\n[2/5] Loading text descriptions ...')
    entity_text_path = os.path.join(script_dir, cfg['entity2text']) if cfg['entity2text'] else None
    relation_text_path = os.path.join(script_dir, cfg['relation2text']) if cfg['relation2text'] else None
    entity_texts = load_text_map(entity_text_path, cfg['text_format'])
    relation_texts = load_text_map(relation_text_path, cfg['text_format'])
    print(f'  Entity texts: {len(entity_texts)}  |  Relation texts: {len(relation_texts)}')

    # ── 加载三元组 ────────────────────────────────────────────────────────────
    train_triplets = load_triplets(os.path.join(data_dir, 'train.txt'))
    print(f'  Train triplets: {len(train_triplets)}')

    # ── 初始化 BERT 和 EDRL 模型 ──────────────────────────────────────────────
    print(f'\n[3/5] Loading BERT from {args.bert_model} ...')
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = EDRLModel(args.bert_model, n_relations, kg_dim=args.kg_dim).to(device)

    # ── 阶段一：对比学习继续预训练 ────────────────────────────────────────────
    if not args.no_pretrain:
        print(f'\n[4a/5] Contrastive Pre-training ({args.pt_epochs} epochs) ...')
        # 构建 BM25 索引（论文 §2.3.2：k1=0.9, b=0.4）
        bm25_index = None
        if HAS_BM25 and relation_texts:
            print('  Building BM25 index (k1=0.9, b=0.4) ...')
            bm25_index, _ = build_bm25_index(
                relation_names, relation_texts,
                k1=args.bm25_k1, b=args.bm25_b
            )
            print(f'  BM25 index built over {len(relation_names)} relations.')
        else:
            print('  BM25 unavailable or no relation texts; using oracle positive.')
        pt_dataset = ContrastiveDataset(
            train_triplets, entity_texts, relation_texts,
            relation_names, n_neg=args.n_neg, bm25=bm25_index
        )
        from functools import partial
        pin = (device.type == 'cuda')
        pt_loader = DataLoader(
            pt_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin,
            collate_fn=partial(contrastive_collate_fn,
                               tokenizer=tokenizer,
                               max_length=args.max_length_pt)
        )
        optimizer_pt = torch.optim.AdamW(model.parameters(), lr=args.lr_pt)
        scaler_pt = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == 'cuda'))

        for epoch in range(args.pt_epochs):
            model.train()
            total_loss = 0.0
            for step, (anchor_enc, pos_enc, neg_enc, n_neg) in enumerate(pt_loader):
                anchor_enc = {k: v.to(device, non_blocking=True) for k, v in anchor_enc.items()}
                pos_enc    = {k: v.to(device, non_blocking=True) for k, v in pos_enc.items()}
                neg_enc    = {k: v.to(device, non_blocking=True) for k, v in neg_enc.items()}

                optimizer_pt.zero_grad()
                with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == 'cuda')):
                    loss = model.forward_contrastive(
                        anchor_enc, pos_enc, neg_enc, n_neg,
                        temperature=args.temperature
                    )
                scaler_pt.scale(loss).backward()
                scaler_pt.step(optimizer_pt)
                scaler_pt.update()
                total_loss += loss.item()

                if (step + 1) % 200 == 0:
                    print(f'  [PT] Epoch {epoch+1}/{args.pt_epochs}  '
                          f'Step {step+1}/{len(pt_loader)}  '
                          f'Loss: {total_loss / (step+1):.4f}')

            print(f'  [PT] Epoch {epoch+1} done. Avg Loss: {total_loss / len(pt_loader):.4f}')

        # 保存预训练权重
        pt_ckpt = os.path.join(script_dir, f'edrl_pretrained_{dataset}.pt')
        torch.save(model.state_dict(), pt_ckpt)
        print(f'  Pre-trained model saved: {pt_ckpt}')
    else:
        print('\n[4a/5] Contrastive pre-training skipped (--no_pretrain).')

    # ── 阶段二：关系分类微调 ──────────────────────────────────────────────────
    print(f'\n[4b/5] Fine-tuning for relation classification ({args.ft_epochs} epochs) ...')
    ft_dataset = FineTuneDataset(train_triplets, entity_texts, relation_name2id)
    from functools import partial
    pin = (device.type == 'cuda')
    ft_loader = DataLoader(
        ft_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=partial(finetune_collate_fn,
                           tokenizer=tokenizer,
                           max_length=args.max_length_ft)
    )
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.lr_ft)
    scaler_ft = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == 'cuda'))

    for epoch in range(args.ft_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for step, (enc, labels) in enumerate(ft_loader):
            enc    = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(device, non_blocking=True)

            optimizer_ft.zero_grad()
            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == 'cuda')):
                loss, logits = model.forward_finetune(enc, labels)
            scaler_ft.scale(loss).backward()
            scaler_ft.step(optimizer_ft)
            scaler_ft.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 200 == 0:
                print(f'  [FT] Epoch {epoch+1}/{args.ft_epochs}  '
                      f'Step {step+1}/{len(ft_loader)}  '
                      f'Loss: {total_loss / (step+1):.4f}  '
                      f'Acc: {correct/total:.4f}')

        print(f'  [FT] Epoch {epoch+1} done. '
              f'Avg Loss: {total_loss / len(ft_loader):.4f}  '
              f'Train Acc: {correct/total:.4f}')

    # 保存微调后权重（BERT*）
    ft_ckpt = os.path.join(script_dir, f'edrl_finetuned_{dataset}.pt')
    torch.save(model.state_dict(), ft_ckpt)
    print(f'  Fine-tuned model saved: {ft_ckpt}')

    # ── 特征提取（公式 2-10）────────────────────────────────────────────────
    print(f'\n[5/5] Extracting entity and relation features (Linear_F output dim={args.kg_dim}) ...')
    model.eval()

    # 实体特征：E_h = Linear_F(BERT*(h))
    entity_text_list = [
        get_text(name, entity_texts, cfg['text_format'])
        for name in entity_names
    ]
    print('  Extracting entity features ...')
    entity_feats = extract_features(
        model, tokenizer, entity_text_list, device,
        batch_size=64, max_length=args.max_length_ext
    )
    entity_out = os.path.join(data_dir, 'entity_features_pretrained.npy')
    np.save(entity_out, entity_feats)
    print(f'  Entity features saved: {entity_out}  shape={entity_feats.shape}')

    # 关系特征：E_r = Linear_F(BERT*(r))
    relation_text_list = [
        get_text(name, relation_texts, cfg['text_format'])
        for name in relation_names
    ]
    print('  Extracting relation features ...')
    relation_feats = extract_features(
        model, tokenizer, relation_text_list, device,
        batch_size=64, max_length=args.max_length_ext
    )
    relation_out = os.path.join(data_dir, 'relation_features_pretrained.npy')
    np.save(relation_out, relation_feats)
    print(f'  Relation features saved: {relation_out}  shape={relation_feats.shape}')

    print('\n✓ EDRL pipeline complete.')
    print(f'  Entity features  → {entity_out}')
    print(f'  Relation features → {relation_out}')
    print(f'  Fine-tuned BERT* → {ft_ckpt}')


if __name__ == '__main__':
    main()

