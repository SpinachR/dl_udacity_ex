from config import Config
from bidirectional_gan.bidirectional_gan_model import BiGAN


def main():
    config = Config(z_dim=100,
                    logdir='./logdir/bi_gan/mnist',
                    sampledir='./sample_image/bi_gan/mnist',
                    model_ckpt='bi_gan.ckpt',
                    checkpoint_basename='./checkpoint/bi_gan')
    model = BiGAN(config)
    model.fit()

if __name__ == '__main__':
    main()