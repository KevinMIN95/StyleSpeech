import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed

import argparse
import os
import json

from models.StyleSpeech import StyleSpeech
from models.Discriminators import Discriminator
from dataloader import prepare_dataloader
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils

def load_checkpoint(checkpoint_path, model, discriminator, G_optim, D_optim, rank, distributed=False):
    assert os.path.isfile(checkpoint_path)
    print("Starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(rank))
    if 'model' in checkpoint_dict:
        if distributed:
            state_dict = {}
            for k,v in checkpoint_dict['model'].items():
                state_dict['module.{}'.format(k)] = v
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint_dict['model'])
        print('Model is loaded!') 
    if 'discriminator' in checkpoint_dict:
        if distributed:
            state_dict = {}
            for k,v in checkpoint_dict['discriminator'].items():
                state_dict['module.{}'.format(k)] = v
            discriminator.load_state_dict(state_dict)
        else:
            discriminator.load_state_dict(checkpoint_dict['discriminator'])
        print('Discriminator is loaded!')
    if 'G_optim' in checkpoint_dict or 'optimizer' in checkpoint_dict:
        if 'optimizer' in checkpoint_dict:
            G_optim.load_state_dict(checkpoint_dict['optimizer'])
        if 'G_optim' in checkpoint_dict:
            G_optim.load_state_dict(checkpoint_dict['G_optim'])
        print('G_optim is loaded!')
    if 'D_optim' in checkpoint_dict:
        D_optim.load_state_dict(checkpoint_dict['D_optim'])
        print('D_optim is loaded!')
    current_step = checkpoint_dict['step'] + 1
    del checkpoint_dict
    return model, discriminator, G_optim, D_optim, current_step


def main(rank, args, c):

    print('Use GPU: {} for training'.format(rank))

    ngpus = args.ngpus
    if args.distributed:
        torch.cuda.set_device(rank % ngpus)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=rank)

    # Define model & loss
    model = StyleSpeech(c).cuda()
    discriminator = Discriminator(c).cuda()
    num_param = utils.get_param_num(model)
    D_num_param = utils.get_param_num(discriminator)

    if rank==0:
        print('Number of Meta-StyleSpeech Parameters:', num_param)
        print('Number of Discriminator Parameters:', D_num_param)
        with open(os.path.join(args.save_path, "model.txt"), "w") as f_log:
            f_log.write(str(model))
            f_log.write(str(discriminator))
        print("Model Has Been Defined")
    
    model_without_ddp = model
    discriminator_without_ddp = discriminator
    if args.distributed:
        c.meta_batch_size = c.meta_batch_size // ngpus
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model_without_ddp = model.module
        discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank])
        discriminator_without_ddp = discriminator.module

    # Optimizer
    G_optim = torch.optim.Adam(model.parameters(), betas=c.betas, eps=c.eps)
    D_optim = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=c.betas, eps=c.eps)
    # Loss
    Loss = model_without_ddp.get_criterion()
    adversarial_loss = discriminator_without_ddp.get_criterion()
    print("Optimizer and Loss Function Defined.")

    # Get dataset
    data_loader = prepare_dataloader(args.data_path, "train.txt", batch_size=c.meta_batch_size, meta_learning=True, seed=rank) 
    print("Data Loader is Prepared")

    # Load checkpoint if exists
    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path)
        model, discriminator, G_optim, D_optim, current_step = load_checkpoint(
                    args.checkpoint_path, model, discriminator, G_optim, D_optim, rank, args.distributed)
        print("\n---Model Restored at Step {}---\n".format(current_step))
    else:
        print("\n---Start New Training---\n")
        current_step = 0
    if rank == 0:
        checkpoint_path = os.path.join(args.save_path, 'ckpt')
        os.makedirs(checkpoint_path, exist_ok=True)
    
    # scheduled optimizer
    G_optim = ScheduledOptim(G_optim, c.decoder_hidden, c.n_warm_up_step, current_step)
    
    # Init logger
    if rank == 0:
        log_path = os.path.join(args.save_path, 'log')
        logger = SummaryWriter(os.path.join(log_path, 'board'))
        with open(os.path.join(log_path, "log.txt"), "a") as f_log:
            f_log.write("Dataset :{}\n Number of Parameters: {}\n".format(c.dataset, num_param))

    # Init synthesis directory
    if rank == 0:
        synth_path = os.path.join(args.save_path, 'synth')
        os.makedirs(synth_path, exist_ok=True)

    model.train()
    while current_step < args.max_iter:
        # Get Training Loader
        for idx, batch in enumerate(data_loader):

            if current_step == args.max_iter:
                break

            losses = {}
            #### Generator ####
            G_optim.zero_grad()
            # Get Support Data
            sid, text, mel_target, D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = model_without_ddp.parse_batch(batch)
            
            # Support Forward
            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _  = model(
                    text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            src_target, _, _ = model_without_ddp.variance_adaptor.length_regulator(src_output, D)

            # Reconstruction loss
            mel_loss, d_loss, f_loss, e_loss = Loss(mel_output,  mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
            losses['G_recon'] = mel_loss
            losses['d_loss'] = d_loss
            losses['f_loss'] = f_loss
            losses['e_loss'] = e_loss
            

            #### META LEARNING ####
            # Get query text 
            B = mel_target.shape[0]
            perm_idx = torch.randperm(B)
            q_text, q_src_len = text[perm_idx], src_len[perm_idx]
            # Generate query speech
            q_mel_output, q_src_output, q_log_duration_output, \
                    _, _, q_src_mask, q_mel_mask, q_mel_len = model_without_ddp.inference(style_vector, q_text, q_src_len)
            # Legulate length of query src
            q_duration_rounded = torch.clamp(torch.round(torch.exp(q_log_duration_output.detach())-1.), min=0)
            q_duration = q_duration_rounded.masked_fill(q_src_mask, 0).long()
            q_src, _, _ = model_without_ddp.variance_adaptor.length_regulator(q_src_output, q_duration)
            # Adverserial loss   
            t_val, s_val, _= discriminator(q_mel_output, q_src, None, sid, q_mel_mask)
            losses['G_GAN_query_t'] = adversarial_loss(t_val, is_real=True)
            losses['G_GAN_query_s'] = adversarial_loss(s_val, is_real=True)

            # Total generator loss
            alpha = 10.0
            G_loss = alpha*losses['G_recon'] + losses['d_loss'] + losses['f_loss'] + losses['e_loss'] + \
                            losses['G_GAN_query_t'] + losses['G_GAN_query_s']
            # Backward loss
            G_loss.backward()
            # Update weights
            G_optim.step_and_update_lr()


            #### Discriminator ####
            D_optim.zero_grad()
            # Real
            real_t_pred, real_s_pred, cls_loss = discriminator(
                                mel_target, src_target.detach(), style_vector.detach(), sid, mask=mel_mask)
            # Fake
            fake_t_pred, fake_s_pred, _ = discriminator(
                                q_mel_output.detach(), q_src.detach(), None, sid, mask=q_mel_mask)
            losses['D_t_loss'] = adversarial_loss(real_t_pred, is_real=True) + adversarial_loss(fake_t_pred, is_real=False)
            losses['D_s_loss'] = adversarial_loss(real_s_pred, is_real=True) + adversarial_loss(fake_s_pred, is_real=False)
            losses['cls_loss'] = cls_loss
            # Total discriminator Loss
            D_loss = losses['D_t_loss'] + losses['D_s_loss'] + losses['cls_loss']
            # Backward
            D_loss.backward()
            # Update weights
            D_optim.step()
            
            # Print log
            if current_step % args.log_step == 0 and current_step != 0 and rank == 0 :
                m_l = losses['G_recon'].item()
                d_l = losses['d_loss'].item()
                f_l = losses['f_loss'].item() 
                e_l = losses['e_loss'].item() 
                g_t_l = losses['G_GAN_query_t'].item()
                g_s_l = losses['G_GAN_query_s'].item()
                d_t_l = losses['D_t_loss'].item() / 2
                d_s_l = losses['D_s_loss'].item() / 2
                cls_l = losses['cls_loss'].item()

                str1 = "Step [{}/{}]:".format(current_step, args.max_iter)
                str2 =  "Mel Loss: {:.4f},\n" \
                        "Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f}\n" \
                        "T G Loss: {:.4f}, T D Loss: {:.4f}, S G Loss: {:.4f}, S D Loss: {:.4f} \n" \
                        "cls_Loss: {:.4f};" \
                        .format(m_l, d_l, f_l, e_l, g_t_l, d_t_l, g_s_l, d_s_l, cls_l)
                print(str1 + "\n" + str2 +"\n")
                with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                    f_log.write(str1 + "\n" + str2 +"\n")
                    
                logger.add_scalar('Train/mel_loss', m_l, current_step)
                logger.add_scalar('Train/duration_loss', d_l, current_step)
                logger.add_scalar('Train/f0_loss', f_l, current_step)
                logger.add_scalar('Train/energy_loss', e_l, current_step)
                logger.add_scalar('Train/G_t_loss', g_t_l, current_step)
                logger.add_scalar('Train/D_t_loss', d_t_l, current_step)
                logger.add_scalar('Train/G_s_loss', g_s_l, current_step)
                logger.add_scalar('Train/D_s_loss', d_s_l, current_step)
                logger.add_scalar('Train/cls_loss', cls_l, current_step)
    
            # Save Checkpoint
            if current_step % args.save_step == 0 and current_step != 0 and rank == 0:
                torch.save({'model': model_without_ddp.state_dict(), 
                            'discriminator': discriminator_without_ddp.state_dict(), 
                            'G_optim': G_optim.state_dict(),'D_optim': D_optim.state_dict(),
                            'step': current_step}, 
                            os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                print("*** Save Checkpoint ***")
                print("Save model at step {}...\n".format(current_step))

            if current_step % args.synth_step == 0 and current_step != 0 and rank == 0:
                length = mel_len[0].item()   
                mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
                mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                q_length = q_mel_len[0].item()
                q_mel = q_mel_output[0, :q_length].detach().cpu().transpose(0, 1)
                # plotting
                utils.plot_data([q_mel.numpy(), mel.numpy(), mel_target.numpy()], 
                    ['Query Spectrogram', 'Recon Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}.png'.format(current_step)))
                print("Synth audios at step {}...\n".format(current_step))
            
            # Evaluate
            if current_step % args.eval_step == 0 and current_step != 0 and rank == 0:
                model.eval()
                with torch.no_grad():
                    m_l, d_l, f_l, e_l = evaluate(args, model_without_ddp, current_step)
                    str_v = "*** Validation ***\n" \
                            "Meta-StyleSpeech Step {},\n" \
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
    
    if rank == 0:
        print("Training Done at Step : {}".format(current_step))
        torch.save({'model': model_without_ddp.state_dict(), 
                    'discriminator': discriminator_without_ddp.state_dict(), 
                    'G_optim': G_optim.state_dict(), 'D_optim': D_optim.state_dict(),
                    'step': current_step}, 
                    os.path.join(checkpoint_path, 'checkpoint_last_{}.pth.tar'.format(current_step)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='dataset/LibriTTS/preprocessed')
    parser.add_argument('--save_path', default='exp_meta_stylespeech')
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--save_step', default=5000, type=int)
    parser.add_argument('--synth_step', default=1000, type=int)
    parser.add_argument('--eval_step', default=5000, type=int)
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pretrained model') 
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='url for setting up distributed training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='distributed backend')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='node rank for distributed training')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True

    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)
    utils.build_env(args.config, 'config.json', args.save_path)

    ngpus = torch.cuda.device_count()
    args.ngpus = ngpus
    args.distributed = ngpus > 1

    if args.distributed:
        args.world_size = ngpus
        mp.spawn(main, nprocs=ngpus, args=(args, config))
    else:
        main(0, args, config)


