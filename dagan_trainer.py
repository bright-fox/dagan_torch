import torch
import time
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import warnings

class DaganTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        dis_optimizer,
        visualizer,
        device="cpu",
        gp_weight=10,
        critic_iterations=5,
    ):
        self.device = device
        self.g = generator.to(device)
        self.g_opt = gen_optimizer
        self.d = discriminator.to(device)
        self.d_opt = dis_optimizer
        self.losses = {"G": [0.0], "D": [0.0], "GP": [0.0], "gradient_norm": [0.0]}
        self.num_steps = 0
        self.epoch = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.visualizer = visualizer

    def _critic_train_iteration(self, x1, x2):
        """
        Train the discriminator
        """
        # Get generated data
        generated_data = self.sample_generator(x1)

        d_real = self.d(x1, x2)
        d_generated = self.d(x1, generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(x1, x2, generated_data)
        self.losses["GP"].append(gradient_penalty.item())

        # Create total loss and optimize
        self.d_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.d_opt.step()

        # Record loss
        self.losses["D"].append(d_loss.item())

    def _generator_train_iteration(self, x1):
        """
        Train the generator
        """
        self.g_opt.zero_grad()

        # Get generated data
        generated_data = self.sample_generator(x1)

        # Calculate loss and optimize
        d_generated = self.d(x1, generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.losses["G"].append(g_loss.item())

    def _gradient_penalty(self, x1, x2, generated_data):
        # Calculate interpolation
        alpha = torch.rand(x1.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(x2).to(self.device)
        interpolated = alpha * x2.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.d(x1, interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(x1.shape[0], -1)
        self.losses["gradient_norm"].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            x1, x2 = data[0].to(torch.float).to(self.device), data[1].to(torch.float).to(self.device)
            self._critic_train_iteration(x1, x2)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(x1)

            self.log_losses()

    def train(self, data_loader, epochs, val_dataloader):
        start_time = int(time.time())

        while self.epoch < epochs:
            print("\nEpoch {}".format(self.epoch))
            print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes\n")

            # log the generations
            self.log_curr_generated_imgs(val_dataloader)
            # train the epoch
            self._train_epoch(data_loader)

            self.epoch += 1
            # self._save_checkpoint()

        # at the end log the generations in table to wandb (hotfix for wandb bug)
        self.visualizer.log_generation()

    def sample_generator(self, input_images, z=None):
        """
        Creates augmentations of the input images with the generator
        """
        if z is None:
            z = torch.randn((input_images.shape[0], self.g.z_dim)).to(self.device)
        return self.g(input_images, z)

    def sample_val_images(self, val_dataloader):
        """
        images have the shape (CxHxW) and have the range [0, 255]
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            idx = torch.randint(0, len(val_dataloader.dataset), (1,))
            return {
                'original': val_dataloader.dataset.originals[idx],
                'augmentation': val_dataloader.dataset.augmentations[idx]
            }

    def log_losses(self):
        if self.num_steps % 100 == 0:
            print(f'[INFO] Iteration {self.num_steps}')
            self.visualizer.log_losses(self.losses)


    def log_curr_generated_imgs(self, val_dataloader):
        """
        Logs the image (original, real augmentation, augmentation of generator) to the visualizer
        """
        val_imgs = self.sample_val_images(val_dataloader)
        real_val_img = ((val_imgs['original'] / 255) - 0.5) / 0.5

        # set generator to eval mode
        self.g.eval()
        with torch.no_grad():
            generated_val_img = self.sample_generator(
                torch.from_numpy(real_val_img)[None, :].to(torch.float).to(self.device)
            )
        # set generator back to training mode
        self.g.train()

        self.visualizer.add_generated_imgs_to_table(
            self.epoch,
            val_imgs['original'],
            val_imgs['augmentation'],
            generated_val_img,
        )
