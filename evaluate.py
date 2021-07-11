import torch
from dataloader import prepare_dataloader


def evaluate(args, model, step):    
    # Get dataset
    data_loader = prepare_dataloader(args.data_path, "val.txt", batch_size=50, shuffle=False) 
 
    # Get loss function
    Loss = model.get_criterion()

    # Evaluation
    mel_l_list = []
    d_l_list = []
    f_l_list = []
    e_l_list = []
    current_step = 0
    for i, batch in enumerate(data_loader):
        # Get Data
        id_ = batch["id"]
        sid, text, mel_target, D, log_D, f0, energy, \
            src_len, mel_len, max_src_len, max_mel_len = model.parse_batch(batch)
    
        with torch.no_grad():
            # Forward
            mel_output, _, _, log_duration_output, f0_output, energy_output, src_mask, mel_mask, out_mel_len = model(
                            text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            # Cal Loss
            mel_loss, d_loss, f_loss, e_loss = Loss(mel_output,  mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)

            # Logger
            m_l = mel_loss.item()
            d_l = d_loss.item()
            f_l = f_loss.item()
            e_l = e_loss.item()

            mel_l_list.append(m_l)
            d_l_list.append(d_l)
            f_l_list.append(f_l)
            e_l_list.append(e_l)

        current_step += 1            
    
    mel_l = sum(mel_l_list) / len(mel_l_list)
    d_l = sum(d_l_list) / len(d_l_list)
    f_l = sum(f_l_list) / len(f_l_list)
    e_l = sum(e_l_list) / len(e_l_list)

    return mel_l, d_l, f_l, e_l


