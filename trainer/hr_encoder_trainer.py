import sys
sys.path.append('/home/sangwon/voice2face')
import os.path as osp
from models.networks import HRImageEncoder
import torch
import torch.nn as nn
import sys
from dataset.lrs import LRS3
import torch.utils.data
import lpips
import wandb
import torchvision
from tools import ImResize_Bicubic, freeze_model


def train(args, ckpt_dir, log_dir):
    
    wandb.init(
        project="EarForEyes",
        config={
        "learning_rate_encoder": 0.0001,
        "architecture": "CNN",
        "dataset": "LRS3",
        "epochs": 1440,
        "batch_size": 15,
        },
        group="HRImageEncoder",
        name=args.exp_name,
        id=args.id,
        resume='allow',
        entity='lsw1240'
    )
    
    
    def validate(test_image):
        with torch.no_grad():
            test_latent = high_res_encoder(test_image)
            test_generated_image = generator(test_latent, None)
            test_loss = criterion_HREncoder(test_generated_image, test_image)
        
        return test_loss
    
    
    # GPU Select
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:2" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    
    # Settings
    save_every_n_epochs = 30
    num_epochs_HREncoder = 1440
    batch_size = 15


    # Load Datasets
    dataset = LRS3()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = LRS3(split='test')
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    

    # Load HR_Encoder
    high_res_encoder = HRImageEncoder().to(device)
    
    # Load Generator
    from stylegan2.dnnlib import util
    from stylegan2 import legacy
    network_pkl = '/home/sangwon/stylegan2-ada-pytorch/training_result/00003-lrs3-auto2-resumeffhq256/network-snapshot-005200.pkl'
    with util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema']
    generator = G.to(device)
    generator.eval()
    freeze_model(generator)
    

    # HR_Encoder Training
    criterion_HREncoder = nn.L1Loss().to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    high_res_encoder_optimizer = torch.optim.Adam(high_res_encoder.parameters(), lr=1e-4)
    
  
    for epoch in range(num_epochs_HREncoder):
        for i, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            image = ImResize_Bicubic(image, 256, antialiasing=True)
            
            high_res_encoder_optimizer.zero_grad()
            latent_vector = high_res_encoder(image)
            
            generated_image = generator(latent_vector, None).to(device)
            loss_vgg = torch.mean(loss_fn_vgg(generated_image, image))
            loss_L1 = criterion_HREncoder(generated_image, image)
            loss = loss_L1 + loss_vgg
            
            loss.backward()
            high_res_encoder_optimizer.step()
            
            wandb.log({"loss": loss.cpu().item()})
            wandb.log({"L1 loss": loss_L1.cpu().item()})
            wandb.log({"Vgg loss": loss_vgg.cpu().item()})
            
        # Validation
        with torch.no_grad():
                for i, test_batch in enumerate(test_dataloader):
                    test_image = test_batch['image'].to(device)
                    test_image = ImResize_Bicubic(test_image, 256, antialiasing=True)
                    break
                val_loss = validate(test_image)
                wandb.log({"val_L1": val_loss})
    
        if (epoch+1) % save_every_n_epochs == 0:
            torch.save(high_res_encoder.state_dict(), osp.join(ckpt_dir, f'{epoch+1:02d}epoch_hr_encoder.pt'))
            image.add_(1).div_(2)
            generated_image.add_(1).div_(2)
            torchvision.utils.save_image(generated_image, osp.join(log_dir, f'{epoch+1:02d}epoch_generated_image.png'))
            torchvision.utils.save_image(image, osp.join(log_dir, f'{epoch+1:02d}epoch_gt_image.png'))
    torch.save(high_res_encoder.state_dict(), osp.join(ckpt_dir, 'last_hr_encoder.pt'))

    wandb.alert(title='끝', text='굳')
    wandb.finish()
