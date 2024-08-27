from .planner import Planner, save_model, sigmoid_focal_loss 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data, ConfusionMatrix
from . import dense_transforms
from torch.utils.data.sampler import SubsetRandomSampler


def train(args):
    from os import path
    model = Planner()

    if args.tensorboard == 'yes':
        train_logger = tb.SummaryWriter('deepnet1/train', flush_secs = 1)
        valid_logger = tb.SummaryWriter('deepnet1/valid', flush_secs = 1)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training == 'yes':
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    # pos_weights = torch.tensor([1200])[None, :, None, None].to(device)
    # det_loss = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weights)
    # det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    aim_loss = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    

    validation_split = 0.2
    random_seed = 42
    dataset_size = 16000
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_data = load_data(dataset_path = "data", 
        batch_size = args.batch, 
        transform = dense_transforms.Compose(
            transforms = [
            # dense_transforms.Resize((150,200)),
            # dense_transforms.ColorJitter(0.2, 0.5, 0.5, 0.2), 
            dense_transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            dense_transforms.RandomHorizontalFlip(), 
            dense_transforms.ToTensor()]), 
        sampler = train_sampler,
        shuffle = False
        )

    
    valid_data = load_data(dataset_path = "data", 
        batch_size = args.batch, 
        sampler = valid_sampler,
        shuffle = False
        )
    

    global_step = 0

    img_size = args.batch * 300 * 400

    for epoch in range(args.num_epoch):
        model.train()
        v_losses = []
        for img, seg, aim in train_data:
            img, seg, aim = img.to(device), seg.to(device).unsqueeze(1).float()/255, aim.to(device)

            # Number of pixels
            n_puck_pixels = torch.sum(seg).item()
            pos_weights = (img_size - n_puck_pixels) / n_puck_pixels

            index_pucks = []
            for i in range(args.batch):
                n_puck_pixels = torch.sum(seg[i, :, :, :]).item()
                if n_puck_pixels >= 10:
                    index_pucks.append(i)

            indices = torch.tensor(index_pucks).to(device)

            # Model outputs
            det_pred, aim_pred = model(img)

            # Segmentation loss
            det_loss_val = sigmoid_focal_loss(inputs = det_pred, targets = seg, pos_weights = pos_weights, alpha = -1, gamma = 2, reduction = 'mean').to(device)

            # L1 Loss - Aimpoint
            aim_pred_include, aim_include = torch.index_select(aim_pred, dim=0, index=indices).to(device), torch.index_select(aim, dim=0, index=indices).to(device)
            aim_loss_val = aim_loss(aim_pred_include, aim_include) * 0.3

            # Total loss
            loss_val = det_loss_val + aim_loss_val 

            if (global_step % 50) == 0:
                print(f"Det loss: {det_loss_val}, Aim loss: {aim_loss_val}, Loss: {loss_val}")

            if args.tensorboard == 'yes':
                # Aim point
                log_seg(train_logger, img, seg, det_pred, global_step)
                log(train_logger, img, aim_include, aim_pred_include, global_step, lw = 15)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        val_conf = ConfusionMatrix()
        for vimg, vseg, vaim in train_data:
            vimg, vseg, vaim = vimg.to(device), vseg.to(device).unsqueeze(1).float()/255, vaim.to(device)

            vdet_pred, vaim_pred = model(vimg)
            v_loss = aim_loss(vaim_pred, vaim)
            v_losses.append(v_loss.detach().cpu().numpy())

            val_conf.add(vdet_pred.argmax(1), vseg)

        if args.tensorboard == 'yes':
            log_seg(valid_logger, vimg, vseg, vdet_pred, global_step)
            log(train_logger, vimg, vaim, vaim_pred, global_step, lw = 15)

        print(f"Epoch {epoch}")

    save_model(model)

def log(logger, img, label, pred, global_step, lw = 15):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0] + 1).cpu().detach().numpy(), 2, ec='g', fill=False, lw=lw))
    ax.add_artist(plt.Circle(WH2*(pred[0] + 1).cpu().detach().numpy(), 2, ec='r', fill=False, lw=lw))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


def log_seg(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    det_final = (torch.sigmoid(det[:4]) > 0.5).long()
    logger.add_images('image', imgs[:4], global_step)
    logger.add_images('label', gt_det[:4], global_step)
    logger.add_images('pred1', det_final, global_step)
    logger.add_images('pred2', torch.sigmoid(det[:4]), global_step)


def log_seg1(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-b', '--batch', default = 128, type = int)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-tb', '--tensorboard', choices = ['yes', 'no'], default = 'no')
    parser.add_argument('-c', '--continue_training', choices = ['yes', 'no'], default = 'no')

    args = parser.parse_args()
    train(args)