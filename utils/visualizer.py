import wandb

class Visualizer():
    def __init__(self, args):
        self.args = args
        self.use_wandb = args.use_wandb
        self.current_epoch = 0
        if self.use_wandb:
            self.wandb_run = wandb.init(project='DAGAN', name=args.name, config=args) if not wandb.run else wandb.run

    def log_generations(self, epoch, original_img, real_augmented_img, generated_augmented_img):
        if not self.use_wandb:
            pass

        table = wandb.Table(
            columns=['Epoch', 'Original', 'Real Augmentation', 'Generated Augmentation'],
            data=[[epoch, wandb.Image(original_img), wandb.Image(real_augmented_img), wandb.Image(generated_augmented_img * 255)]],
        )
        self.wandb_run.log({'Generations_Per_Epoch': table})

    def log_losses(self, losses):
        # print losses to console
        print("D: {}".format(losses["D"][-1]))
        print("Raw D: {}".format(losses["D"][-1] - losses["GP"][-1]))
        print("GP: {}".format(losses["GP"][-1]))
        print("Gradient norm: {}".format(losses["gradient_norm"][-1]))
        print("G: {}".format(self.losses["G"][-1]))

        # log losses to wandb
        if self.use_wandb:
            self.wandb_run.log({
                'D': losses['D'][-1],
                'Raw D': losses['D'][-1] - losses['GP'][-1],
                'GP': losses['GP'][-1],
                'Gradient norm': losses['gradient_norm'][-1],
                'G': losses['G'][-1],
            })

