from config import Config
from bidirectional_gan.ali import aliGAN


def main():
    config = Config(z_dim=256,
                    x_height=32,
                    x_width=32,
                    x_depth=3,

                    batch_size=100,
                    epoch=100,
                    sample_every_n_steps=2,
                    summary_every_n_steps=2,
                    savemodel_every_n_steps=2,

                    logdir='./svhn/logdir/ali_gan',
                    sampledir='./svhn/sample_image/ali_gan',
                    model_ckpt='ali_gan.ckpt',
                    checkpoint_basename='./svhn/checkpoint/ali_gan')
    model = aliGAN(config)
    model.train()

if __name__ == '__main__':
    main()