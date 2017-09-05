class Config(object):
    def __init__(self, **kwargs):
        # configuration for building the network
        self.seed_data = kwargs.get("seed_data", 1)
        self.seed = kwargs.get("seed", 1)

        self.x_dim = kwargs.get("x_dim", 784)
        self.x_height = kwargs.get("x_height", 28)
        self.x_width = kwargs.get("x_width", 28)
        self.x_depth = kwargs.get("x_depth", 1)
        self.image_channel = kwargs.get("image_channel", 1)

        self.y_dim = kwargs.get("y_dim", 10)   # label dim
        self.cat_dim = kwargs.get("cat_dim", 10)  # categorical variable dim

        self.c_dim = kwargs.get("c_dim", 2)   # continous variables dim
        self.z_dim = kwargs.get("z_dim", 64)   # z (noise) dim

        self.epoch = kwargs.get("epoch", 10)
        self.batch_size = kwargs.get("batch_size", 150)

        self.unlabeled_batch = kwargs.get("unlabeled_batch", 100)
        self.num_of_labeled_data_for_each_classes = kwargs.get("num_of_labeled_data_for_each_classes", 10)
        self.num_of_classes = kwargs.get("num_of_classes", 10)
        self.labeled_batch = kwargs.get("labeled_batch", 28)  # deprecated

        self.lr = kwargs.get("lr", 0.0001)
        self.lr_d = kwargs.get("lr_d", 0.0001)
        self.lr_g = kwargs.get("lr_g", 0.0001)

        self.beta1 = kwargs.get("beta1", 0.5)  # recommend value in dcgan paper
        self.beta2 = kwargs.get("beta2", 0.001)

        self.logdir = kwargs.get("logdir", "./logdir/infogan/mnist")
        self.sampledir = kwargs.get("sampledir", "./sample_image/mnist/infogan")

        self.max_steps = kwargs.get("max_steps", 100000)
        self.sample_every_n_steps = kwargs.get("sample_every_n_steps", 5)
        self.summary_every_n_steps = kwargs.get("summary_every_n_steps", 5)
        self.savemodel_every_n_steps = kwargs.get("savemodel_every_n_steps", 5)
        self.save_model_secs = kwargs.get("save_model_secs", 1200)

        self.load_ckpt = kwargs.get("load_ckpt", False)
        self.model_ckpt = kwargs.get("model_ckpt", "cgan.ckpt")
        self.checkpoint_basename = kwargs.get("checkpoint_basename", "./checkpoint/infogan")