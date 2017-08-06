from config import Config
from info_gan import info_gan_model


def main():
    config = Config(batch_size=64,
                    logdir='./logdir/infogan/mnist',
                    sampledir='./sample_image/mnist/infogan',
                    model_ckpt='infogan.ckpt',
                    checkpoint_basename='./checkpoint/infogan')
    model = info_gan_model.InfoGAN(config)
    model.fit()

if __name__ == '__main__':
    main()