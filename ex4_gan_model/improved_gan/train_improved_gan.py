from config import Config
from improved_gan.improved_gan_model import ImprovedGAN


def main():
    config = Config(unlabeled_batch=100,
                    num_of_labeled_data_for_each_classes=10,
                    num_of_classes=10,
                    z_dim=100,
                    logdir='./logdir/improved_gan/mnist',
                    sampledir='./sample_image/improved_gan/mnist',
                    model_ckpt='improved_gan.ckpt',
                    checkpoint_basename='./checkpoint/improved_gan')
    model = ImprovedGAN(config)
    model.train()

if __name__ == '__main__':
    main()