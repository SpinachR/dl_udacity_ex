from config import Config
import conditional_gan as cgan


def main():
    config = Config()
    model = cgan.ConditionalGAN(config)
    model.fit()

if __name__ == '__main__':
    main()