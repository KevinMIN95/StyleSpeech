import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from models.StyleSpeech import StyleSpeech
from dataloader import prepare_dataloader
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path)
    if 'model' in checkpoint_dict:
        model.load_state_dict(checkpoint_dict['model'])
        print('Model is loaded!')
    if 'optimizer' in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        print('Optimizer is loaded!')
    current_step = checkpoint_dict['step'] + 1
    return model, optimizer, current_step


def main(args, c):

    # Define model
    model = StyleSpeech(c).cuda()
    print("StyleSpeech Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of StyleSpeech Parameters:', num_param)
    with open(os.path.join(args.save_path, "model.txt"), "w") as f_log:
        f_log.write(str(model))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas=c.betas, eps=c.eps)
    # Loss
    Loss = model.get_criterion()
    print("Optimizer and Loss Function Defined.")

    # Get dataset
    data_loader = prepare_dataloader(args.data_path, "train.txt", shuffle=True, batch_size=c.batch_size) 
    print("Data Loader is Prepared.")

    # Load checkpoint if exists
    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path)
        model, optimizer, current_step= load_checkpoint(args.checkpoint_path, model, optimizer)
        print("\n---Model Restored at Step {}---\n".format(current_step))
    else:
        print("\n---Start New Training---\n")
        current_step = 0
    checkpoint_path = os.path.join(args.save_path, 'ckpt')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Scheduled optimizer
    scheduled_optim = ScheduledOptim(optimizer, c.decoder_hidden, c.n_warm_up_step, current_step)

    # Init logger
    log_path = os.path.join(args.save_path, 'log')
    logger = SummaryWriter(os.path.join(log_path, 'board'))
    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
        f_log.write("Dataset :{}\n Number of Parameters: {}\n".format(c.dataset, num_param))

    # Init synthesis directory
    synth_path = os.path.join(args.save_path, 'synth')
    os.makedirs(synth_path, exist_ok=True)

    # Training
    model.train()
    while current_step < args.max_iter:        
        # Get Training Loader
        for idx, batch in enumerate(data_loader):

            if current_step == args.max_iter:
                break
                
            # Get Data
            sid, text, mel_target, D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = model.parse_batch(batch)
                
            # Forward
            scheduled_optim.zero_grad()
            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _  = model(
                    text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)

            mel_loss, d_loss, f_loss, e_loss = Loss(mel_output, mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)

            # Total loss
            total_loss = mel_loss + d_loss + f_loss + e_loss
            # Backward
            total_loss.backward()
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip_thresh)
            # Update weights
            scheduled_optim.step_and_update_lr()

            # Print log
            if current_step % args.log_step == 0 and current_step != 0:    
                t_l = total_loss.item()
                m_l = mel_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()

                str1 = "Step [{}/{}]:".format(current_step, args.max_iter)
                str2 = "Total Loss: {:.4f}\nMel Loss: {:.4f},\n" \
                        "Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f} ;" \
                        .format(t_l, m_l, d_l, f_l, e_l)
                print(str1 + "\n" + str2 +"\n")
                with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                    f_log.write(str1 + "\n" + str2 +"\n")

                logger.add_scalar('Train/total_loss', t_l, current_step)
                logger.add_scalar('Train/mel_loss', m_l, current_step)
                logger.add_scalar('Train/duration_loss', d_l, current_step)
                logger.add_scalar('Train/f0_loss', f_l, current_step)
                logger.add_scalar('Train/energy_loss', e_l, current_step)
            
            # Save Checkpoint
            if current_step % args.save_step == 0 and current_step != 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': current_step}, 
                    os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                print("*** Save Checkpoint ***")
                print("Save model at step {}...\n".format(current_step))

            if current_step % args.synth_step == 0 and current_step != 0:
                length = mel_len[0].item()
                mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
                mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                # plotting
                utils.plot_data([mel.numpy(), mel_target.numpy()], 
                    ['Synthesized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}.png'.format(current_step)))
                print("Synth spectrograms at step {}...\n".format(current_step))
            
            if current_step % args.eval_step == 0 and current_step != 0:
                model.eval()
                with torch.no_grad():
                    m_l, d_l, f_l, e_l = evaluate(args, model, current_step)
                    str_v = "*** Validation ***\n" \
                            "StyleSpeech Step {},\n" \
                            "Mel Loss: {}\nDuration Loss:{}\nF0 Loss: {}\nEnergy Loss: {}" \
                            .format(current_step, m_l, d_l, f_l, e_l)
                    print(str_v + "\n" )
                    with open(os.path.join(log_path, "eval.txt"), "a") as f_log:
                        f_log.write(str_v + "\n")
                    logger.add_scalar('Validation/mel_loss', m_l, current_step)
                    logger.add_scalar('Validation/duration_loss', d_l, current_step)
                    logger.add_scalar('Validation/f0_loss', f_l, current_step)
                    logger.add_scalar('Validation/energy_loss', e_l, current_step)
                model.train()

            current_step += 1 

    print("Training Done at Step : {}".format(current_step))
    torch.save({'model': model.state_dict(), 'optimizer': scheduled_optim.state_dict(), 'step': current_step}, 
                os.path.join(checkpoint_path, 'checkpoint_last_{}.pth.tar'.format(current_step)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='dataset/LibriTTS/preprocessed')
    parser.add_argument('--save_path', default='exp_stylespeech')
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--save_step', default=5000, type=int)
    parser.add_argument('--synth_step', default=1000, type=int)
    parser.add_argument('--eval_step', default=5000, type=int)
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the pretrained model') 

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True

    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)
    utils.build_env(args.config, 'config.json', args.save_path)

    main(args, config)
