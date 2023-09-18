import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

class Visualizer(object):
    def __init__(self, config):
        self.config = config


    def visualize(self, epoch, dataloader, model):
        """
        1.  generate images
        2. generate reconstructed images
        3. visualize
        :param epoch:
        :param dataloader:
        :param model:
        :return:
        """

        #generate images from the data
        dataiter = iter(dataloader)
        x, label = next(dataiter)
        x, label = x.cuda(), label.cuda()


        with torch.no_grad():
            if self.config.experiment == 'mnist_vae':
                x = x.reshape(self.config.batch_size, 1, -1)  # flatten x
                y = model(x)
                x, y = x.reshape(self.config.batch_size, 1, 28, 28), y.reshape(self.config.batch_size, 1, 28, 28)
            else:
                y = model(x)



        x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()



        images_original =[x[i,0] for i in range(10)]
        images_reconstructed =[y[i,0] for i in range(10)]

        # for i, (img1, img2) in enumerate(zip(images_original, images_reconstructed)):
        #     assert isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray), f"Image {i} is not a numpy array."
        #     assert img1.shape == img2.shape, f"Image {i} from list1 and list2 do not have the same shape."

        # Concatenate each pair of images along width and then stack them vertically

        fig, axes = plt.subplots(nrows=10, ncols=2,
                                 figsize=(6, 30))  # 10 rows for 10 image pairs, 2 columns for two images in each pair

        # Use a loop to display each image pair
        concatenated = []
        for i in range(10):
            axes[i, 0].imshow(images_original[i], cmap='gray', norm='linear')
            axes[i, 1].imshow(images_reconstructed[i], cmap='gray', norm='linear')

            concatenated.append(np.concatenate([images_original[i], images_reconstructed[i]], axis=0))

            # Remove axis ticks for better visualization
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

        plt.tight_layout()  # Adjusts spacing between subplots for better layout
        # plt.show()

        concatenated = np.concatenate(concatenated, axis=1)
        if epoch == 'final':
            wandb.log({"images": wandb.Image(concatenated)})
            return

        # concatenated_images = np.vstack([np.hstack([img1, img2]) for img1, img2 in zip(images_original, images_reconstructed)])
        #
        # # Display the resulting image grid
        # plt.imshow(concatenated_images)

        filename = f'/epoch_{epoch}'
        path = self.config.dir + filename
        plt.savefig(path)





