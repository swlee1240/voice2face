from models.networks import HRImageEncoder, SpectrogramEncoder
import sys
sys.path.append('/home/sangwon/voice2face')
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from dataset.lrs import LRS3
import torch.utils.data
from tools import ImResize_Bicubic, freeze_model, InfoNCE_with_L2
import wandb
import torchvision.transforms
from stylegan2.dnnlib import util
import os
import os.path as osp


def train(args, ckpt_dir, log_dir):
    
    wandb.init(
        project="EarForEyes",
        config={
        "learning_rate_encoder": 0.0001,
        "architecture": "CNN",
        "dataset": "LRS3",
        "epochs": 720,
        "batch_size": 80
        },
        group="AudioEncoder",
        name=args.exp_name,
        id=args.id,
        resume='allow',
        entity='lsw1240'
    )
    
    
    def test(audio):
        with torch.no_grad():
            latent = audio_encoder(audio)
            image_g = generator(latent, None)
        
        return image_g
    
    
    # GPU Select
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:2" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    save_every_n_epochs = 30
    num_epochs_AuEncoder = 720
    batch_size = 80
    
    
    # Load Dataset
    dataset = LRS3()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    
    # Load Audio_Encoder
    audio_encoder = SpectrogramEncoder().to(device)
    
    
    # Load HR_Encoder
    high_res_encoder = HRImageEncoder().to(device)
    checkpoint_hr = torch.load('/home/sangwon/pytorch_ear_for_face/checkpoint/hr_1e-4_norm/540epoch_hr_encoder.pt', map_location=device)
    high_res_encoder.load_state_dict(checkpoint_hr)
    high_res_encoder.eval()
    freeze_model(high_res_encoder)
    
    
    # Load Generator
    from stylegan2 import legacy
    network_pkl = '/home/sangwon/stylegan2-ada-pytorch/training_result/00003-lrs3-auto2-resumeffhq256/network-snapshot-005200.pkl'
    with util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema']
    generator = G.to(device)
    generator.eval()
    freeze_model(generator)
    
    horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
    
    
    #AudioEncoder Training
    criterion_L1 = nn.L1Loss().to(device)
    criterion_infoNCE = InfoNCE_with_L2(device)
    audio_encoder_optimizer = torch.optim.Adam(audio_encoder.parameters(), lr=1e-4)
    
    # Weight for InfoNCE
    L = 2

    for epoch in range(num_epochs_AuEncoder):
        for i, batch in enumerate(dataloader):
            audio = batch['audio'].to(device)
            audio_path = batch['audio_path']
            image = batch['image'].to(device)
            image = ImResize_Bicubic(image, 256, antialiasing=True)
        
            audio_encoder_optimizer.zero_grad()
            
            latent_vector_audio = audio_encoder(audio)
            with torch.no_grad():
                fliped_image = horizontal_flip(image)
                img_latent = high_res_encoder(image)
                flip_latent = high_res_encoder(fliped_image)
                avg_latent = (img_latent + flip_latent) / 2
            loss_L1 = criterion_L1(latent_vector_audio, avg_latent)
            loss_infoNCE = criterion_infoNCE.loss_fn(latent_vector_audio, avg_latent) 
            loss = loss_L1 + L * loss_infoNCE
            
            loss.backward()
            audio_encoder_optimizer.step()
            
            wandb.log({"loss (L1)": loss_L1.cpu().item()})
            wandb.log({"loss (InfoNCE)": loss_infoNCE.cpu().item()})
            wandb.log({"loss (L1 + InfoNCE)": loss.cpu().item()})
                
        if (epoch+1) % save_every_n_epochs == 0:
            torch.save(audio_encoder.state_dict(), osp.join(ckpt_dir, f'{epoch+1:02d}epoch_au_encoder.pt'))
            file_path = log_dir+'/audio_path.txt'
            with open(file_path,'a+') as file:
                file.write('\n')
                file.write(f'epoch{epoch+1} : {audio_path}')
            with torch.no_grad():
                image_test = test(audio).cpu()
                image.add_(1).div_(2)
                image_test.add_(1).div_(2)
            torchvision.utils.save_image(image, osp.join(log_dir, f'{epoch+1:02d}epoch_gt_image.png'))
            torchvision.utils.save_image(image_test, osp.join(log_dir, f'{epoch+1:02d}epoch_generated_image.png'))
    torch.save(audio_encoder.state_dict(), osp.join(ckpt_dir, 'last_au_encoder.pt'))
        
        
    wandb.alert(title='끝', text='굳')
    wandb.finish()