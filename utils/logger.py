import wandb
import numpy as np

class Logger():
    """
    Logger is used to log information about the training.
    - losses
    - evaluation on noisy environments
    """

    def __init__(self, args, wandb_project='DAGAN'):
        self.args = args
        self.current_epoch = 0
        self.wandb_run = wandb.init(project=wandb_project) if args.sweep else wandb.init(project=wandb_project, name=args.name, config=args)
        self.eval_imgs_table = wandb.Table(columns=['Epoch', 'Original', 'Real Augmentation', 'Generated Augmentation'])

    def log_eval_imgs(self, epoch, original, real_aug, gen_aug):
        """
        Log the original image, real augmentation and generated augmentation into a wandb table
        """
        # transform the original and real augmentation image of
        # shape (channel, height, width) to be digested by wandb.Image
        o = original.transpose(1, 2, 0)
        aug = real_aug.transpose(1, 2, 0)

        # transform the tensor of the generated image of
        # shape (batch, channel, height, width) to be digested by wandb.Image
        g = (gen_aug * 0.5 + 0.5) * 255
        g = np.squeeze(g.detach().cpu().numpy()).transpose(1,2,0)
        
        self.eval_imgs_table.add_data(epoch, wandb.Image(o), wandb.Image(aug), wandb.Image(g))

    def upload_eval_imgs(self):
        """
        Log the images to wandb
        """
        self.wandb_run.log({'Generations': self.eval_imgs_table})

    def log_losses(self, losses):
        # log losses to wandb
        self.wandb_run.log({
            'D': losses['D'][-1],
            'Raw D': losses['D'][-1] - losses['GP'][-1],
            'GP': losses['GP'][-1],
            'Gradient norm': losses['gradient_norm'][-1],
            'G': losses['G'][-1],
        })

