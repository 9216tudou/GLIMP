import itertools
import os
from collections import Counter
import dgl
import numpy as np
# from gensim.models import Word2Vec
# from rdkit import Chem
# from rdkit.Chem import MACCSkeys
import torch
from dgl.nn.pytorch import EdgeWeightNorm
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
# import forgi
# import forgi.graph.bulge_graph as fgb
import torch.nn.functional as F

nt_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'I': 0, 'H': 1, 'M': 2, 'S': 3}
rev_dict = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A', 'A': 'U', 'U': 'A'}
nts = ['A', 'C', 'G', 'T']
DPCP = {'AA': [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'AT': [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0,
                   0.76847598772376],
            'AG': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'AC': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'TA': [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257, 0.3059118434042811,
                   0.32686549630327577],
            'TT': [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'TG': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'TC': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GA': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GT': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'GG': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332],
            'GC': [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165, 0.6780274730118172,
                   0.8400043540595654],
            'CA': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'CT': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'CG': [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134, 0.5170412957708868,
                   0.32686549630327577],
            'CC': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332]}



def pse_normalize(pse_knc):
    mu = np.mean(pse_knc, axis=0)
    sigma = np.std(pse_knc, axis=0)
    return (pse_knc - mu) / sigma


def do_CL(X, Y, T):
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)

    criterion = torch.nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, T)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    return CL_loss, CL_acc


def dual_CL(X, Y, T=0.3):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, T)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, T)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


def pad_batch(bg, max_input_len):
    num_batch = bg.batch_size
    graphs = dgl.unbatch(bg)
    h_node = bg.ndata['h']
    max_num_nodes = max_input_len
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()
    for i, g in enumerate(graphs):
        num_node = g.num_nodes()
        padded_h_node[-num_node:, i] = g.ndata['h']
        src_padding_mask[i, :max_num_nodes - num_node] = True

    return padded_h_node, src_padding_mask


def get_mask(bg):
    """
    :param bg:
    :return: batch_size * num_node  Ture if the degree is zero
    """
    num_batch = bg.batch_size
    in_d = bg.in_degrees() > 0
    o_d = bg.out_degrees() > 0
    mask = ~in_d & ~o_d
    mask = mask.view(num_batch, -1).to(bg.device)
    return mask


def kmer2num(kmer):
    ans = 0
    for i in range(len(kmer)):
        d = nt_dict[kmer[i]]
        ans = d + ans * 4
    return ans


def de_Bruijn_graph(seq, k):
    """
    :param seq:序列表示
    :param k: kmer的k
    :return:
    """
    seq2id = []  # id seq

    num_node = 1 << 2 * k
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        n_id = kmer2num(kmer)
        seq2id.append(n_id)

    edge_freq = Counter(list(zip(seq2id[:-1], seq2id[1:])))
    graph = dgl.graph(list(edge_freq.keys()), num_nodes=num_node)
    graph.ndata['mask'] = torch.LongTensor(range(num_node))
    weight = torch.FloatTensor(list(edge_freq.values()))
    norm = EdgeWeightNorm(norm='both')
    norm_weight = norm(graph, weight)
    graph.edata['weight'] = norm_weight
    return graph


def get_neighbor(k):
    """

    :param k:
    :return: dict {kmer1:[neighbor1,neighbor2,...],kmer2:[...],...}
    """
    pools = itertools.product(['0', '1', '2', '3'], repeat=k)
    ans = {}
    for kmer in pools:
        ans[int(''.join(kmer), 4)] = []
        for i in range(len(kmer)):
            for j in range(1, 4):
                new_kmer = list(kmer)
                new_kmer[i] = str((int(kmer[i]) + j) % 4)
                ans[int(''.join(kmer), 4)].append(int(''.join(new_kmer), 4))
        ans[int(''.join(kmer), 4)].append(int(''.join(kmer), 4))
    return ans


def idng(seq, k, neighbor_dict, window_size=20):
    """
    create an interval directed neighbor-based graph
    """

    def get_intersect(l1, l2):
        res = []
        p1 = 0
        p2 = 0
        while p1 < len(l1) and p2 < len(l2):
            if l1[p1] < l2[p2]:
                p1 += 1
            elif l1[p1] > l2[p2]:
                p2 += 1
            else:
                res.append(l1[p1])
                p1 += 1
                p2 += 1
        return res

    src = []
    dst = []
    num_node = 1 << 2 * k
    for i in range(len(seq) - k + 1):
        cur_kmer = seq[i:i + k]
        window = []
        for j in range(i + 1, i + window_size + 1):
            if j + k > len(seq):
                break
            window.append(kmer2num(seq[j:j + k]))
        neighbor = neighbor_dict[kmer2num(cur_kmer)]
        tmp_dst = get_intersect(sorted(neighbor), sorted(window))
        tmp_src = [kmer2num(cur_kmer)] * len(tmp_dst)
        src.extend(tmp_src)
        dst.extend(tmp_dst)
    edge_freq = Counter(list(zip(src, dst)))
    graph = dgl.graph(list(edge_freq.keys()), num_nodes=num_node)
    graph.ndata['mask'] = torch.LongTensor(range(num_node))
    weight = torch.FloatTensor(list(edge_freq.values()))
    norm = EdgeWeightNorm(norm='both')
    norm_weight = norm(graph, weight)
    graph.edata['weight'] = norm_weight
    return graph


# def idng(seq, k, neighbor_dict, window_size=20, similarity_threshold=0.6, alpha=0.5):
#     """
#     构建改进后的IDNG图：
#     - 顶点集合 V = { v_i = S[i:i+k] | i=1,...,|S|-k+1 }
#     - 第一类边 E1：连接相邻的k-mer，权重为 alpha + (1-alpha)*sim(v_i, v_{i+1})
#     - 第二类边 E2：对于满足 2 <= j-i <= window_size 且 sim(v_i,v_j) >= similarity_threshold 的k-mer对，
#       权重为 (1-alpha)*sim(v_i, v_j)
#     参数:
#         seq: 启动子序列字符串
#         k: k-mer长度
#         neighbor_dict: 预计算的字典，将k-mer数值映射到其邻居集合
#         window_size: 窗口大小
#         similarity_threshold: 相似性阈值 t
#         alpha: 连续性信息平衡参数
#     返回:
#         dgl图对象，图中边数据 'weight' 为归一化后的权重
#     """
#     def get_intersect(l1, l2):
#         res = []
#         p1 = 0
#         p2 = 0
#         while p1 < len(l1) and p2 < len(l2):
#             if l1[p1] < l2[p2]:
#                 p1 += 1
#             elif l1[p1] > l2[p2]:
#                 p2 += 1
#             else:
#                 res.append(l1[p1])
#                 p1 += 1
#                 p2 += 1
#         return res
#
#     def default_sim(kmer1, kmer2):
#         # 使用简单匹配比例作为相似性度量
#         matches = sum(1 for a, b in zip(kmer1, kmer2) if a == b)
#         return matches / len(kmer1)
#
#     src = []
#     dst = []
#     weights = []
#     num_node = 1 << (2 * k)  # 节点数取决于k-mer总数（假定全空间）
#
#     # 构建E1边：相邻k-mer
#     for i in range(len(seq) - k):
#         cur_kmer = seq[i:i+k]
#         next_kmer = seq[i+1:i+1+k]
#         cur_num = kmer2num(cur_kmer)
#         next_num = kmer2num(next_kmer)
#         sim_val = default_sim(cur_kmer, next_kmer)
#         w_val = alpha + (1 - alpha) * sim_val
#         src.append(cur_num)
#         dst.append(next_num)
#         weights.append(w_val)
#
#     # 构建E2边：非连续但在窗口内且相似性满足阈值的k-mer对
#     for i in range(len(seq) - k + 1):
#         cur_kmer = seq[i:i+k]
#         cur_num = kmer2num(cur_kmer)
#         # 只考虑与当前位置相距至少2且不超过window_size的k-mer
#         for j in range(i + 2, min(i + window_size + 1, len(seq) - k + 1)):
#             cand_kmer = seq[j:j+k]
#             cand_num = kmer2num(cand_kmer)
#             # 若候选k-mer在预先定义的邻居集合中
#             if cand_num not in neighbor_dict[cur_num]:
#                 continue
#             sim_val = default_sim(cur_kmer, cand_kmer)
#             if sim_val >= similarity_threshold:
#                 w_val = (1 - alpha) * sim_val
#                 src.append(cur_num)
#                 dst.append(cand_num)
#                 weights.append(w_val)
#
#     # 累加相同边的权重
#     edge_dict = {}
#     for s, d, w in zip(src, dst, weights):
#         edge_dict[(s, d)] = edge_dict.get((s, d), 0) + w
#
#     final_src = [s for (s, d) in edge_dict.keys()]
#     final_dst = [d for (s, d) in edge_dict.keys()]
#     final_weights = list(edge_dict.values())
#
#     # 构建图
#     graph = dgl.graph((final_src, final_dst), num_nodes=num_node)
#     graph.ndata['mask'] = torch.LongTensor(range(num_node))
#     weight_tensor = torch.FloatTensor(final_weights)
#     norm = EdgeWeightNorm(norm='both')
#     norm_weight = norm(graph, weight_tensor)
#     graph.edata['weight'] = norm_weight
#     return graph
def default_sim(kmer1, kmer2):
    """
    默认相似性函数，计算两个k-mer中相同字符的比例。
    """
    matches = sum(1 for a, b in zip(kmer1, kmer2) if a == b)
    return matches / len(kmer1)

def num2kmer(i, k):
    kmer = ''
    nt2int = ['A', 'C', 'G', 'T']
    while k > 0:
        kmer = nt2int[int(i % 4)] + kmer
        i /= 4
        k -= 1
    return kmer


def generate_weight_by_shapelet(seq, shapelet_info, k=3):  # 1
    """
    :param seq: seq of rna
    :param shapelet_info: shapelet,score,tag
    :return:
    """

    ans = [0] * (1 << 2 * k)
    n = len(seq)
    for x, y in zip(range(n), range(k, n)):
        cur_word = seq[x:y]
        for shapelet_seq, score, _ in shapelet_info:
            if cur_word in shapelet_seq and ans[kmer2num(cur_word)] == 0:
                ans[kmer2num(cur_word)] += score
    return ans
#
#
# def generate_weight_by_shapelet(seq, shapelet_info, k=3):
#     """
#     :param seq: seq of RNA
#     :param shapelet_info: list of shapelet sequences with their scores and tags
#     :param k: k-mer length
#     :param DPCP: dictionary of DPCP properties for dinucleotides
#     :return: list of weights for each k-mer
#     """
#     # Initialize the result list for storing weights, using size based on k-mer length
#     ans = [0] * (1 << 2 * k)
#     n = len(seq)
#
#     for x in range(n - k + 1):
#         cur_word = seq[x:x + k]
#         cur_score = 0
#
#         # Iterate over all shapelets in shapelet_info
#         for shapelet_seq, base_score, _ in shapelet_info:
#             if cur_word == shapelet_seq:
#                 # Add the base score of the shapelet sequence
#                 cur_score = max(cur_score, base_score)
#
#                 # Add DPCP-based weight if DPCP is provided
#                 if DPCP:
#                     DPCP_score = sum(DPCP.get(cur_word[i:i + 2], [0] * 6) for i in range(len(cur_word) - 1))
#                     cur_score += sum(DPCP_score)  # Sum DPCP weights to enhance the score
#
#                 # After processing the current shapelet, update the corresponding k-mer weight
#                 ans[kmer2num(cur_word)] += cur_score
#
#     return ans

def get_freq(seq):
    """
    2, 3, 4 ,5 mer 1360d
    kmer freq vector
    :param seq:
    :return:
    """
    freq = []

    for k in range(1, 6):
        tmp = torch.FloatTensor([0] * (1 << 2 * k))
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            tmp[kmer2num(kmer)] += 1
        tmp /= (len(seq) - k + 1)
        freq.extend(tmp)

    return torch.FloatTensor(freq)


def get_rev(seq):
    """
    get the rev complement
    :param seq:
    :return:
    """
    return ''.join([rev_dict[seq[i]] for i in range(len(seq) - 1, -1, -1)])


def make_path(path):
    try:
        if os.path.exists(path):
            return
        if '/' not in path:
            os.makedirs(path)
            return
        make_path(os.path.dirname(path))
        os.makedirs(path)
    except Exception as e:
        print(f"创建目录 {path} 时发生错误：{e}")

def random_pseudo_rev(seq):
    return ''.join([nts[np.random.randint(4)] for _ in range(len(seq))])


def eval_output(model_perf, path):
    with open(os.path.join(path, f"Evaluate_Result_TestSet.txt"), 'w') as f:
        f.write("AUROC=%s\tAUPRC=%s\tAccuracy=%s\tMCC=%s\tRecall=%s\tPrecision=%s\tf1_score=%s\n" %
                (model_perf["auroc"], model_perf["auprc"], model_perf["accuracy"], model_perf["mcc"],
                 model_perf["recall"], model_perf["precision"], model_perf["f1"]))
        f.write("\n######NOTE#######\n")
        f.write(
            "#According to help_documentation of sklearn.metrics.classification_report:in binary classification, recall of the positive class is also known as sensitivity; recall of the negative class is specificity#\n\n")
        f.write(model_perf["class_report"])


# Evaluate performance of model
def evaluate_performance(y_test, y_pred, y_prob):
    # AUROC
    auroc = metrics.roc_auc_score(y_test, y_prob)
    auroc_curve = metrics.roc_curve(y_test, y_prob)
    # AUPRC
    auprc = metrics.average_precision_score(y_test, y_prob)
    auprc_curve = metrics.precision_recall_curve(y_test, y_prob)
    # Accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # MCC
    mcc = metrics.matthews_corrcoef(y_test, y_pred)

    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    class_report = metrics.classification_report(y_test, y_pred, target_names=["control", "case"])

    model_perf = {"auroc": auroc, "auroc_curve": auroc_curve,
                  "auprc": auprc, "auprc_curve": auprc_curve,
                  "accuracy": accuracy, "mcc": mcc,
                  "recall": recall, "precision": precision, "f1": f1,
                  "class_report": class_report}
    return model_perf


# Plot AUROC of model
def plot_AUROC(model_perf, path):
    """
    生成 AUROC 曲线，符合学术论文格式
    """
    # 获取 AUROC 和 FPR、TPR
    roc_auc = model_perf["auroc"]
    fpr, tpr, _ = model_perf["auroc_curve"]

    # 存储 AUROC 数据
    pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(os.path.join(path, "AUROC_info_randomly_generated.txt"),
                                                  header=True, index=False, sep='\t')

    # 绘制 AUROC 曲线
    plt.figure(figsize=(6, 6))  # 论文标准大小
    lw = 3
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUROC (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("AUROC Curve", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle="--")

    # 保存为高质量 PDF
    plt.savefig(os.path.join(path, "AUROC_TestSet.pdf"), format="pdf", dpi=300)
    plt.close()


def plot_training_curves(history, path):
    """
    训练过程曲线绘制：
    - Loss 下降曲线
    - AUROC, AUPRC, ACC, MCC, F1 指标变化曲线
    """
    epochs = history["epoch"]

    plt.figure(figsize=(12, 6))

    # ✅ 在绘图时转换为 NumPy，确保数据格式正确
    loss_values = np.array(history["loss"]) if isinstance(history["loss"][0], torch.Tensor) else history["loss"]

    # 1️⃣ Loss 下降曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='r', lw=2, label="Loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training Loss", fontsize=16)
    plt.grid(True, linestyle="--")
    plt.legend(fontsize=12)

    # 2️⃣ 评估指标变化曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["auroc"], marker='o', linestyle='-', color='b', lw=2, label="AUROC")
    plt.plot(epochs, history["auprc"], marker='s', linestyle='-', color='g', lw=2, label="AUPRC")
    plt.plot(epochs, history["acc"], marker='^', linestyle='-', color='purple', lw=2, label="Accuracy")
    plt.plot(epochs, history["mcc"], marker='v', linestyle='-', color='orange', lw=2, label="MCC")
    plt.plot(epochs, history["f1"], marker='d', linestyle='-', color='cyan', lw=2, label="F1-score")

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("Model Performance Metrics", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--")

    # 保存图像（高质量 PDF 格式）
    plt.tight_layout()
    plt.savefig(os.path.join(path, "Training_Results.pdf"), format="pdf", dpi=300)
    plt.close()
