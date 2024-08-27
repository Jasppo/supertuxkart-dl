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


    det_loss = torch.nn.BCEWithLogitsLoss()
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

    for epoch in range(args.num_epoch):
        model.train()
        t_acc_50, t_acc_70, t_acc_90 = [], [], []
        v_acc_50, v_acc_70, v_acc_90 = [], [], []

        for img, seg, lbl, aim in train_data:
            img, seg, lbl, aim = img.to(device), seg.to(device).unsqueeze(1).float()/255, lbl.unsqueeze(1).float().to(device), aim.to(device)

            index_pucks = []
            for i in range(args.batch):
                label = lbl[i]
                if label == 1:
                    index_pucks.append(i)

            indices = torch.tensor(index_pucks).to(device)

            # Model outputs
            det_pred, aim_pred = model(img)

            # Detection loss
            det_loss_val = det_loss(det_pred, lbl).to(device)

            # L1 Loss - Aimpoint
            aim_pred_include, aim_include = torch.index_select(aim_pred, dim=0, index=indices).to(device), torch.index_select(aim, dim=0, index=indices).to(device)
            aim_loss_val = aim_loss(aim_pred_include, aim_include) * 0.3

            # Total loss
            loss_val = det_loss_val + aim_loss_val 

            t_acc_50.append(accuracy(det_pred, lbl, 0.5))
            t_acc_70.append(accuracy(det_pred, lbl, 0.7))
            t_acc_90.append(accuracy(det_pred, lbl, 0.9))


            if (global_step % 50) == 0:
                print(f"Det loss: {det_loss_val}, Aim loss: {aim_loss_val}, Loss: {loss_val}")

            if args.tensorboard == 'yes':
                # Aim point
                log(train_logger, img, aim, aim_pred, global_step, lw = 15)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        print(f"Train 50: {np.mean(t_acc_50)}, Train 70: {np.mean(t_acc_70)}, Train 90: {np.mean(t_acc_90)}")

        model.eval()
        val_confusion = ConfusionMatrix(1)
        for vimg, vseg, vlbl, vaim in train_data:
            vimg, vseg, vlbl, vaim = vimg.to(device), vseg.to(device).unsqueeze(1).float()/255, vlbl.unsqueeze(1).float().to(device), vaim.to(device)

            vdet_pred, vaim_pred = model(vimg)

            v_acc_50.append(accuracy(vdet_pred, vlbl, 0.5))
            v_acc_70.append(accuracy(vdet_pred, vlbl, 0.7))
            v_acc_90.append(accuracy(vdet_pred, vlbl, 0.9))

        if args.tensorboard == 'yes':
            log(valid_logger, vimg, vaim, vaim_pred, global_step, lw = 15)

        print(f"Valid 50: {np.mean(v_acc_50)}, Valid 70: {np.mean(v_acc_70)}, Valid 90: {np.mean(v_acc_90)}")

        print(f"Epoch {epoch}")

    save_model(model)

def accuracy(pred, gt, decision_threshold):
    pred = (torch.sigmoid(pred) > decision_threshold).long().view(-1, 1)
    compare = (pred == gt).long()
    accuracy = (torch.sum(compare) / args.batch).item()
    return(accuracy)

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