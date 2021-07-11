import torch
import torch.nn as nn
from models.Modules import LinearNorm, ConvNorm, get_sinusoid_encoding_table
import utils


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """
    def __init__(self, config):
        super(VarianceAdaptor, self).__init__()

        self.hidden_dim = config.variance_predictor_filter_size
        self.predictor_kernel_size = config.variance_predictor_kernel_size
        self.embedding_kernel_size = config.variance_embedding_kernel_size
        self.dropout = config.variance_dropout

        # Duration
        self.duration_predictor = VariancePredictor(self.hidden_dim, self.hidden_dim,
                                                            self.predictor_kernel_size, dropout=self.dropout)
        # Pitch
        self.pitch_predictor = VariancePredictor(self.hidden_dim, self.hidden_dim, self.predictor_kernel_size, 
                                                            dropout=self.dropout)
        self.pitch_embedding = VarianceEmbedding(1, self.hidden_dim, self.embedding_kernel_size, self.dropout)
        # Energy
        self.energy_predictor = VariancePredictor(self.hidden_dim, self.hidden_dim, self.predictor_kernel_size, 
                                                            dropout=self.dropout)
        self.energy_embedding = VarianceEmbedding(1, self.hidden_dim, self.embedding_kernel_size, self.dropout)
        # Phoneme
        self.ln = nn.LayerNorm(self.hidden_dim)

        # Length regulator
        self.length_regulator = LengthRegulator(self.hidden_dim, config.max_seq_len)
    
    def forward(self, x, src_mask, mel_len=None, mel_mask=None, 
                        duration_target=None, pitch_target=None, energy_target=None, max_len=None):
        # Duration
        log_duration_prediction = self.duration_predictor(x, src_mask)

        # Pitch & Energy 
        pitch_prediction = self.pitch_predictor(x, src_mask) 
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(pitch_target.unsqueeze(-1))
        else:
            pitch_embedding = self.pitch_embedding(pitch_prediction.unsqueeze(-1))

        energy_prediction = self.energy_predictor(x, src_mask) 
        if energy_target is not None:
            energy_embedding = self.energy_embedding(energy_target.unsqueeze(-1))
        else:
            energy_embedding = self.energy_embedding(energy_prediction.unsqueeze(-1))

        x = self.ln(x) + pitch_embedding + energy_embedding

        # Length regulate
        if duration_target is not None:
            output, pe, mel_len = self.length_regulator(x, duration_target, max_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction)-1.0), min=0)            
            duration_rounded = duration_rounded.masked_fill(src_mask, 0).long()
            output, pe, mel_len = self.length_regulator(x, duration_rounded)
            mel_mask = utils.get_mask_from_lengths(mel_len)

        # Phoneme-wise positional encoding
        output = output + pe
        return output, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask


class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self, hidden_size, max_pos):
        super(LengthRegulator, self).__init__()
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(max_pos+1, hidden_size), requires_grad=False)

    def LR(self, x, duration, max_len):
        output = list()
        position = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded, pos = self.expand(batch, expand_target)
            output.append(expanded)
            position.append(pos)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
            position = utils.pad(position, max_len)
        else:
            output = utils.pad(output)
            position = utils.pad(position)
        return output, position, torch.LongTensor(mel_len).cuda()

    def expand(self, batch, predicted):
        out = list()
        pos = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
            pos.append(self.position_enc[:expand_size, :])
        out = torch.cat(out, 0)
        pos = torch.cat(pos, 0)
        return out, pos

    def forward(self, x, duration, max_len=None):
        output, position, mel_len = self.LR(x, duration, max_len)
        return output, position, mel_len
        

class VariancePredictor(nn.Module):
    """ Variance Predictor """
    def __init__(self, input_size, filter_size, kernel_size, output_size=1, n_layers=2, dropout=0.5):
        super(VariancePredictor, self).__init__()

        convs = [ConvNorm(input_size, filter_size, kernel_size)]
        for _ in range(n_layers-1):
            convs.append(ConvNorm(filter_size, filter_size, kernel_size))
        self.convs = nn.ModuleList(convs)
        self.lns = nn.ModuleList([nn.LayerNorm(filter_size) for _ in range(n_layers)])
        self.linear_layer = nn.Linear(filter_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        for conv, ln in zip(self.convs, self.lns):
            x = x.transpose(1,2)
            x = self.relu(conv(x))
            x = x.transpose(1,2)
            x = ln(x)
            x = self.dropout(x)

        out = self.linear_layer(x)

        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0)
        return out.squeeze(-1)


class VarianceEmbedding(nn.Module):
    """ Variance Embedding """
    def __init__(self, input_size, embed_size, kernel_size, dropout):
        super(VarianceEmbedding, self).__init__()
        self.conv1 = ConvNorm(input_size, embed_size, kernel_size)
        self.conv2 = ConvNorm(embed_size, embed_size, kernel_size)
        self.fc = LinearNorm(embed_size, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x.transpose(1,2)

        out = self.dropout(self.fc(x))
        return out