import wandb
import numpy as np

class Visualizer():
    """
    Visualizer is used to log information about the training.
    - losses are logged in console and wandb
    - evaluation images are logged in wandb
    """

    def __init__(self, args, wandb_project='DAGAN'):
        self.args = args
        self.current_epoch = 0
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            self.wandb_run = wandb.init(project=wandb_project, name=args.name, config=args) if not wandb.run else wandb.run
            self.val_images_table = wandb.Table(columns=['Epoch', 'Original', 'Real Augmentation', 'Generated Augmentation'])

    def add_imgs_to_table(self, epoch, original, real_aug, gen_aug):
        """
        Add the original image, real augmentation and generated augmentation into a wandb table
        """

        if not self.use_wandb:
            pass

        # transform the original and real augmentation image of
        # shape (channel, height, width) to be digested by wandb.Image
        o = original.transpose(1, 2, 0)
        aug = real_aug.transpose(1, 2, 0)

        # transform the tensor of the generated image of
        # shape (batch, channel, height, width) to be digested by wandb.Image
        g = (gen_aug * 0.5 + 0.5) * 255
        g = np.squeeze(g.detach().cpu().numpy()).transpose(1,2,0)
        
        self.val_images_table.add_data(epoch, wandb.Image(o), wandb.Image(aug), wandb.Image(g))

    def log_generation(self):
        """
        Log the images to wandb
        """
        if not self.use_wandb:
            pass

        self.wandb_run.log({'Generations': self.val_images_table})

    def log_losses(self, losses):
        # print losses to console
        print("D: {}".format(losses["D"][-1]))
        print("Raw D: {}".format(losses["D"][-1] - losses["GP"][-1]))
        print("GP: {}".format(losses["GP"][-1]))
        print("Gradient norm: {}".format(losses["gradient_norm"][-1]))
        print("G: {}".format(losses["G"][-1]))

        # log losses to wandb
        if self.use_wandb:
            self.wandb_run.log({
                'D': losses['D'][-1],
                'Raw D': losses['D'][-1] - losses['GP'][-1],
                'GP': losses['GP'][-1],
                'Gradient norm': losses['gradient_norm'][-1],
                'G': losses['G'][-1],
            })

