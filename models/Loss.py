import torch
import torch.nn as nn


class StyleSpeechLoss(nn.Module):
    """ StyleSpeech Loss """
    def __init__(self):
        super(StyleSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, mel, mel_target, log_d_predicted, log_d_target, 
                        p_predicted, p_target, e_predicted, e_target, src_len, mel_len):
        B = mel_target.shape[0]
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False

        mel_loss = 0.
        d_loss = 0.
        p_loss = 0.
        e_loss = 0.

        for b, (mel_l, src_l) in enumerate(zip(mel_len, src_len)):
            mel_loss += self.mae_loss(mel[b, :mel_l, :], mel_target[b, :mel_l, :])
            d_loss += self.mse_loss(log_d_predicted[b, :src_l], log_d_target[b, :src_l])
            p_loss += self.mse_loss(p_predicted[b, :src_l], p_target[b, :src_l])
            e_loss += self.mse_loss(e_predicted[b, :src_l], e_target[b, :src_l])

        mel_loss = mel_loss / B
        d_loss = d_loss / B
        p_loss = p_loss / B
        e_loss = e_loss / B

        return mel_loss, d_loss, p_loss, e_loss


class LSGANLoss(nn.Module):
    """ LSGAN Loss """
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, r, is_real):
        if is_real: 
            ones = torch.ones(r.size(), requires_grad=False).to(r.device)
            loss = self.criterion(r, ones)
        else:
            zeros = torch.zeros(r.size(), requires_grad=False).to(r.device)
            loss = self.criterion(r, zeros)
        return loss