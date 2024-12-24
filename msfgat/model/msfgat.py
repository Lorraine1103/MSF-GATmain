from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import RDKFingerprint
from msfgat.data import GetPubChemFPs, create_graph, get_atom_features_dim
import csv

# mol2vec
import seaborn as sns
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
from gensim.models import word2vec

model = word2vec.Word2Vec.load('model_300dim.pkl')

atts_out = []


class SQ(nn.Module):
    def __init__(self, args):
        super(SQ, self).__init__()
        self.fp_2_dim = args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size
        self.args = args
        if hasattr(args, 'fp_type'):
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'

        if self.fp_type == 'mixed':
            self.fp_dim = 6311
        else:
            self.fp_dim = 1024

        if hasattr(args, 'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None

        self.fc1_fp = nn.Linear(self.fp_dim, self.fp_2_dim)

        self.act_func = nn.ReLU()

        self.fc2_fp = nn.Linear(self.fp_2_dim, self.hidden_dim)

        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smile):
        fp_list = []
        vec_list = []
        for i, one in enumerate(smile):
            fp = []
            vec = []
            mol = Chem.MolFromSmiles(one)

            if self.fp_type == 'mixed':
                # 1 MACCS 167
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp.extend(list(fp_maccs))

                # 2 Morgan 2048
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fp.extend(list(fp_morgan))

                # 3 TT 2048
                fp_TT = AllChem.GetHashedTopologicalTorsionFingerprint(mol)
                fp.extend(list(fp_TT))

                # 4-pp
                fp_4pp = RDKFingerprint(mol, maxPath=4)
                fp.extend(list(fp_4pp))

            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)

            # mol2vec
            sentence = mol2alt_sentence(mol, 1)
            sen = sentences2vec([sentence], model, unseen='UNK')
            sen_list = sen.tolist()
            vec.extend(sen_list[0])
            vec_list.append(vec)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:, self.fp_changebit - 1] = np.ones(fp_list[:, self.fp_changebit - 1].shape)
            fp_list.tolist()

        fp_list = torch.Tensor(fp_list)
        vec_list = torch.Tensor(vec_list)

        if self.cuda:
            fp_list = fp_list.cuda()
            vec_list = vec_list.cuda()

        # FP
        fpn_out = self.fc1_fp(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2_fp(fpn_out)

        # mol2vec
        vec_out = vec_list

        sq_out = torch.cat([fpn_out, vec_out], axis=1)

        return sq_out


class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, dropout_gnn, alpha, inter_graph, concat=True):
        super(GATv2Layer, self).__init__()
        self.dropout_gnn = dropout_gnn
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        self.inter_graph = inter_graph

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # 添加一个线性变换层
        self.b = nn.Linear(2 * out_features, 1, bias=False)  # 这是计算注意力系数的新线性层

    def forward(self, mole_out, adj):
        Wh = torch.mm(mole_out, self.W)  # 节点特征的线性变换
        N = Wh.size(0)

        # 首先应用非线性变换，再计算注意力系数
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(a_input)
        e = self.b(e).squeeze(2)  # 使用新增的线性层b计算注意力系数

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, Wh)

        if self.concat:
            return nn.functional.elu(output)
        else:
            return output




class GATv2One(nn.Module):
    def __init__(self, args):
        super(GATv2One, self).__init__()
        self.nfeat = get_atom_features_dim()
        self.nhid = args.nhid
        self.dropout_gnn = args.dropout_gat
        self.atom_dim = args.hidden_size
        self.alpha = 0.2
        self.nheads = args.nheads
        self.args = args
        self.dropout = nn.Dropout(p=self.dropout_gnn)

        if hasattr(args, 'inter_graph'):
            self.inter_graph = args.inter_graph
        else:
            self.inter_graph = None

        self.attentions = [GATv2Layer(self.nfeat, self.nhid, dropout_gnn=self.dropout_gnn, alpha=self.alpha,
                                    inter_graph=self.inter_graph, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATv2Layer(self.nhid * self.nheads, self.atom_dim, dropout_gnn=self.dropout_gnn, alpha=self.alpha,
                                inter_graph=self.inter_graph, concat=False)

    def forward(self, mole_out, adj):
        mole_out = self.dropout(mole_out)
        mole_out = torch.cat([att(mole_out, adj) for att in self.attentions], dim=1)
        mole_out = self.dropout(mole_out)
        mole_out = nn.functional.elu(self.out_att(mole_out, adj))
        return nn.functional.log_softmax(mole_out, dim=1)



class GATv2Encoder(nn.Module):
    def __init__(self, args):
        super(GATv2Encoder, self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.encoder = GATv2One(self.args)

    def forward(self, mols, smiles):
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()

        gat_outs = []
        for i, one in enumerate(smiles):
            adj = []
            mol = Chem.MolFromSmiles(one)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj / 1
            adj = torch.from_numpy(adj)
            if self.cuda:
                adj = adj.cuda()

            atom_start, atom_size = atom_index[i]
            one_feature = atom_feature[atom_start:atom_start + atom_size]

            gat_atoms_out = self.encoder(one_feature, adj)
            gat_out = gat_atoms_out.sum(dim=0) / atom_size
            gat_outs.append(gat_out)
        gat_outs = torch.stack(gat_outs, dim=0)
        return gat_outs


class GATv2(nn.Module):
    def __init__(self, args):
        super(GATv2, self).__init__()
        self.args = args
        self.encoder = GATv2Encoder(self.args)

    def forward(self, smile):
        mol = create_graph(smile, self.args)
        gat_out = self.encoder.forward(mol, smile)

        return gat_out


class MsfgatModel(nn.Module):
    def __init__(self, is_classif, gat_scale, cuda, dropout_sq):
        super(MsfgatModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_sq = dropout_sq
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self, args):
        self.encoder3 = GATv2(args)

    def create_sq(self, args):
        self.encoder2 = SQ(args)

    def create_scale(self, args):
        linear_dim = int(args.hidden_size)
        if self.gat_scale == 1:
            self.gru_gat = nn.GRU(linear_dim, linear_dim, num_layers=1, batch_first=True)
        elif self.gat_scale == 0:
            self.gru_fpn = nn.GRU(linear_dim * 2, linear_dim * 2, num_layers=1, batch_first=True)
        else:
            self.gat_dim = int((linear_dim * 2 * self.gat_scale) // 1)
            self.gru_gat = nn.GRU(linear_dim, self.gat_dim, num_layers=1, batch_first=True)
            self.gru_fpn = nn.GRU(linear_dim * 2, linear_dim * 2 - self.gat_dim, num_layers=1, batch_first=True)

        self.act_func = nn.ReLU()

    def create_ffn(self, args):
        linear_dim = args.hidden_size
        if self.gat_scale == 1:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_sq),
                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_sq),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )
        elif self.gat_scale == 0:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_sq),
                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_sq),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )

        else:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_sq),
                nn.Linear(in_features=linear_dim * 2, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_sq),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )

    def forward(self, input):
        if self.gat_scale == 1:
            gat_out = self.encoder3(input)
            gat_out = gat_out.unsqueeze(1)  # (batch_size, 1, linear_dim)
            gat_out, _ = self.gru_gat(gat_out)
            gat_out = gat_out.squeeze(1)  # (batch_size, linear_dim)
            output = self.act_func(gat_out)
        elif self.gat_scale == 0:
            fpn_out = self.encoder2(input)
            fpn_out = fpn_out.unsqueeze(1)  # (batch_size, 1, linear_dim)
            fpn_out, _ = self.gru_fpn(fpn_out)
            fpn_out = fpn_out.squeeze(1)  # (batch_size, linear_dim)
            output = self.act_func(fpn_out)
        else:
            gat_out = self.encoder3(input)
            fpn_out = self.encoder2(input)

            gat_out = gat_out.unsqueeze(1)  # (batch_size, 1, linear_dim)
            fpn_out = fpn_out.unsqueeze(1)  # (batch_size, 1, linear_dim)

            gat_out, _ = self.gru_gat(gat_out)
            gat_out = gat_out.squeeze(1)  # (batch_size, gat_dim)

            fpn_out, _ = self.gru_fpn(fpn_out)
            fpn_out = fpn_out.squeeze(1)  # (batch_size, linear_dim * 2 - gat_dim)

            output = torch.cat([gat_out, fpn_out], axis=1)  # (batch_size, linear_dim * 2)
            output = self.act_func(output)

        output = self.ffn(output)

        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        return output


def get_atts_out():
    return atts_out


def MSFGAT(args):
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = MsfgatModel(is_classif, args.gat_scale, args.cuda, args.dropout)
    if args.gat_scale == 1:
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_sq(args)
        model.create_ffn(args)
    else:
        model.create_gat(args)
        model.create_sq(args)
        model.create_scale(args)
        model.create_ffn(args)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model