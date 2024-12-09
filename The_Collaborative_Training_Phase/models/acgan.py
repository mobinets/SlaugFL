import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.BasicModule import BasicModule
from models.base import BaseModel


class Generator_strong(nn.Module):
    def __init__(self, n_classes, image_size, channels=3, latent_dim=100):
        super(Generator_strong, self).__init__()
        self.label_emb = nn.Embedding(n_classes, latent_dim)

        # first linear layer
        self.fc1 = nn.Linear(latent_dim, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
                nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
                nn.BatchNorm2d(192),
                nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
                nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
                nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
                nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
                nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        fc1 = self.fc1(gen_input)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        return tconv5

    def get_params(self):
        return self.state_dict()

    def set_params(self, model_params):
        self.load_state_dict(model_params)


class Discriminator_strong(nn.Module):
    def __init__(self, n_classes, image_size, channels=1):
        super(Discriminator_strong, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
                nn.Conv2d(256, 512, 3, 1, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4 * 4 * 512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4 * 4 * 512, n_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        conv1 = self.conv1(img)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 4 * 4 * 512)
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)

        return realfake, classes

    def get_params(self):
        return self.state_dict()

    def set_params(self, model_params):
        self.load_state_dict(model_params)


class Generator_weak(BasicModule):
    def __init__(self, device, n_classes, image_size, channels=3, noise_dim=100):
        super(Generator_weak, self).__init__()
        self.model_name = 'Generator'
        self.device = device
        self.n_classes = n_classes
        self.noise_dim = noise_dim
        self.init_size = image_size // 4  
        self.input_emb = nn.Sequential(nn.Linear(self.n_classes, 4096))
        self.noise_emb = nn.Sequential(nn.Linear(self.noise_dim, 4096))

        self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, channels, 3, stride=1, padding=1),
                nn.Tanh(),
        )

    def forward(self, noise,labels):
        one_hot_labels = self.get_one_hot(labels,self.n_classes)
        embedded_input = self.input_emb(one_hot_labels)
        embedded_noise = self.noise_emb(noise)
        z = torch.cat((embedded_noise, embedded_input), dim=1)
        z = z.view(z.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(z)
        return img

    def get_one_hot(self, target, num_class):
        one_hot=torch.zeros(target.shape[0],num_class).cuda(self.device)
        one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
        return one_hot

    def get_params(self):
        return self.state_dict()

    def set_params(self, model_params):
        self.load_state_dict(model_params)

class Discriminator_weak(nn.Module):
    def __init__(self, n_classes, image_size, channels=1):
        super(Discriminator_weak, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
                *discriminator_block(channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = math.ceil(image_size / 2 ** 4)

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

    def get_params(self):
        return self.state_dict()

    def set_params(self, model_params):
        self.load_state_dict(model_params)


default_opt = {
    "lr":         0.0002,  # adam: learning rate
    "b1":         0.5,  # adam: decay of first order momentum of gradient
    "b2":         0.999,  # adam: decay of first order momentum of gradient
}


class Model(BaseModel):
    """
    AC-GAN model
    :param dataset_info: tuple of dataset info (num_classes, image_size, channels)
    :param optimizer: a model optimizer (This optimizer is ignored)
    :param device:  model device
    """

    def __init__(self, num_classes, image_size, channels, optimizer,args, device=None):
        super(Model, self).__init__(num_classes, None, device)
        self.size = 0
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        if args.model_outputdim == 10 and args.GAN_type==0:
            self.latent_dim = 100
        elif args.model_outputdim == 10 and args.GAN_type==1:
            self.latent_dim = 110
        elif args.model_outputdim == 100 and args.GAN_type==0:
            self.latent_dim = 256

        self.device = 'cpu' if not device else device

        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()
        # Initialize generator and discriminator
        if args.GAN_type==0:
            self.generator = Generator_weak(self.device,num_classes, image_size, channels, self.latent_dim)
            self.discriminator = Discriminator_weak(num_classes, image_size, channels)
        elif args.GAN_type==1:
            self.generator = Generator_strong(num_classes, image_size, channels, self.latent_dim)
            self.discriminator = Discriminator_strong(num_classes, image_size, channels)

        self._init_models()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=default_opt["lr"],
                                            betas=(default_opt["b1"], default_opt["b2"]))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=default_opt["lr"],
                                            betas=(default_opt["b1"], default_opt["b2"]))

        self.FloatTensor = torch.cuda.FloatTensor if self.device else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.device else torch.LongTensor

    @staticmethod
    def _weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def _init_models(self):
        """
        Initialize model weights
        :return:
        """
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.adversarial_loss.to(self.device)
        self.auxiliary_loss.to(self.device)

        # Initialize weights
        self.generator.apply(self._weights_init_normal)
        self.discriminator.apply(self._weights_init_normal)

    def _sample_image(self, n_row, image_name, image_save_dir="cifar10-image-0.2-12000-2"):
        """
        generating image data for debug. generating images and saving image file
        :param n_row: indicate the number of generate image data (will generate n_row*n_row images )
        :param image_name:  output image name
        :param image_save_dir: output image save dir
        :return:
        """
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        if not os.path.isdir(image_save_dir):
            os.makedirs(image_save_dir, exist_ok=True)

        # Sample noise
        z = Variable(self.FloatTensor(np.random.normal(0, 1, (n_row * 10, self.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(10) for num in range(n_row)])
        labels = Variable(self.LongTensor(labels))
        gen_imgs = self.generator(z, labels)
        save_image(gen_imgs.data, os.path.join(image_save_dir, "%d.png" % image_name), nrow=n_row, normalize=True)

    def create_model(self):
        pass

    def forward(self, x):
        raise NotImplemented("ACGAN not support forward inference yet.")

    def get_params(self):
        return self.generator.get_params(), self.discriminator.get_params()

    def set_params(self, model_params):
        generator_params, discriminator_params = model_params
        self.generator.set_params(generator_params)
        self.discriminator.set_params(discriminator_params)

    def get_gradients(self, data, model_len):
        raise NotImplemented("GAN mode get gradients method is not implemented yet")

    def solve_inner(self, data, num_epochs=1, batch_size=64, sample_interval=-1, verbose=True,model_name=None,gan_model_save_dir=None,class_models=None, anchor=None,):
        """
        Solves local optimization problem
        :param data:
        :param num_epochs:
        :param batch_size:
        :param sample_interval:
        :param verbose:
        :return: (soln, comp)
            - soln: local optimization solution
            - comp: number of FLOPs executed in training process
        """
        self.generator.train()
        self.discriminator.train()
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        if self.args.GAN_type==1:
            ranger = range(num_epochs)
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,pin_memory=True)
            for epoch in ranger:
                for i, (imgs, labels) in enumerate(data_loader):
                    batch_size = imgs.shape[0]

                    # Adversarial ground truths
                    valid = Variable(self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                    fake = Variable(self.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                    # Configure input
                    real_imgs = Variable(imgs.type(self.FloatTensor))
                    labels = Variable(labels.type(self.LongTensor))

                    # -----------------
                    #  Train Generator
                    # -----------------
                    self.optimizer_G.zero_grad()

                    # Sample noise and labels as generator input
                    z = Variable(self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                    gen_labels = Variable(self.LongTensor(np.random.randint(0, self.num_classes, batch_size)))

                    # Generate a batch of images
                    gen_imgs = self.generator(z, gen_labels)
                    
                    # Loss measures generator's ability to fool the discriminator
                    validity, pred_label = self.discriminator(gen_imgs)
                    valid=valid.squeeze(-1)
                    g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))

                    g_loss.backward()
                    self.optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    self.optimizer_D.zero_grad()

                    # Loss for real images
                    real_pred, real_aux = self.discriminator(real_imgs)
                    d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

                    # Loss for fake images
                    fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
                    fake = fake.squeeze(-1)
                    d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

                    # Total discriminator loss
                    d_loss = (d_real_loss + d_fake_loss) / 2

                    # Calculate discriminator accuracy
                    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                    gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                    d_loss.backward()
                    self.optimizer_D.step()

                    if verbose:
                        print(
                                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                                % (epoch, num_epochs, i, len(data_loader), d_loss.item(), 100 * d_acc, g_loss.item())
                        )

                    batches_done = epoch * len(data_loader) + i
                    if sample_interval > 0 and batches_done % sample_interval == 0:
                        self._sample_image(n_row=len(data.classes), image_name=batches_done)

        elif self.args.GAN_type==0:
            ranger = range(num_epochs)
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,pin_memory=True)
            for epoch in ranger:
                for i, (imgs, labels) in enumerate(data_loader):
                    batch_size = imgs.shape[0]
            
                    # Adversarial ground truths
                    valid = Variable(self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                    fake = Variable(self.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)


                    # -----------------
                    #  Train Generator
                    # -----------------
                    self.optimizer_G.zero_grad()
                    z = Variable(self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                    gen_labels = Variable(self.LongTensor(np.random.randint(0, self.num_classes, batch_size)))

                    gen_imgs = self.generator(z, gen_labels)

                    #get feature
                    feature = [[] for i in range(self.num_classes)]
                    for imgi in range(len(gen_imgs)):
                        label = gen_labels[imgi]
                        output,feature_i,_,_,_,_ = class_models[label](gen_imgs[imgi].unsqueeze(0),out_feature = True)
                        feature[label].append(feature_i)
                    class_mem = torch.zeros(self.num_classes).to(self.device)
                    feature_now =[] 
                    for label in range(self.num_classes):
                        if feature[label] != []:
                            class_mem[label] = True
                            tmp = (torch.mean(torch.cat(feature[label],dim=0),dim=0)).unsqueeze(0)
                            feature_now.append(tmp)
                    feature_now = torch.cat(feature_now,dim=0)
                    feature_anchor = anchor[class_mem]

                    # ##10-class data
                    # loss1 = self.weak_loss(feature_now,feature_anchor)
                    # z1, z2 = torch.split(z, z.size(0)//2, dim=0)
                    # gen_img1, gen_img2 = torch.split(gen_imgs, z1.size(0), dim=0)
                    # lz = torch.mean(torch.abs(gen_img1 - gen_img2)) / torch.mean(torch.abs(z1 - z2))
                    # eps = 1 * 1e-5
                    # loss2 = 5000/(lz + eps)
                    # g_loss = (loss1 + loss2)

                    ##100-class data
                    loss1 = self.weak_loss(feature_now,feature_anchor)*0.01
                    z1, z2 = torch.split(z, z.size(0)//2, dim=0)
                    gen_img1, gen_img2 = torch.split(gen_imgs, z1.size(0), dim=0)
                    lz = torch.mean(torch.abs(gen_img1 - gen_img2)) / torch.mean(torch.abs(z1 - z2))
                    eps = 1 * 1e-5
                    loss2 = 100/(lz + eps)
                    g_loss = 0.5 * (loss1 + loss2)

                    g_loss.backward()
                    self.optimizer_G.step()

                    if verbose:
                        print(
                                "[Epoch %d/%d] [Batch %d/%d]  [G loss1: %f, div_loss: %f]"
                                % (epoch, num_epochs, i, len(data_loader), loss1.item(),  loss2.item())
                        )

                    batches_done = epoch * len(data_loader) + i
                    if sample_interval > 0 and (batches_done+1) % 156 == 0:
                        self._sample_image(n_row=len(data.classes), image_name=batches_done)
               
        solution = self.get_params()

        comp = 0  # compute cost
        return solution, comp

    def solve_iters(self, data, num_iters=1, batch_size=64):
        """
        Solves local optimization problem
        :param data:
        :param num_iters:
        :param batch_size:
        :return:
        """
        raise NotImplemented("GAN mode solve iter method is not implemented yet")

    def test(self, test_sets):
        self.generator.eval()
        self.discriminator.eval()
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        test_g_loss = []
        test_d_loss = []
        d_correct = 0
        batch_size = 1000
        with torch.no_grad():
            data_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=True)
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                # Adversarial ground truths
                valid = Variable(self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(data.type(self.FloatTensor))
                labels = Variable(target.type(self.LongTensor))

                # -----------------
                #  Generator
                # -----------------
                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                gen_labels = Variable(self.LongTensor(np.random.randint(0, self.num_classes, batch_size)))

                # Generate a batch of images
                gen_imgs = self.generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = self.discriminator(gen_imgs)
                g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))
                test_g_loss.append(g_loss.item())

                # ---------------------
                #  Discriminator
                # ---------------------
                # Loss for real images
                real_pred, real_aux = self.discriminator(real_imgs)
                d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                test_d_loss.append(d_loss.item())

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.sum(np.argmax(pred, axis=1) == gt)
                d_correct += d_acc

        test_g_loss = np.mean(np.array(test_g_loss))
        test_d_loss = np.mean(np.array(test_d_loss))

        return d_correct, (test_g_loss, test_d_loss)

    def save(self, name, model_save_dir="ACGAN_model"):
        """
        save model parameters
        :param name: model parameter file name
        :param model_save_dir: output dir
        :return:
        """
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)

        generator_path = os.path.join(model_save_dir, "{}-generator.pt".format(name))
        discriminator_path = os.path.join(model_save_dir, "{}-discriminator.pt".format(name))
        torch.save(self.generator.get_params(), generator_path)
        torch.save(self.discriminator.get_params(), discriminator_path)

    def load_model(self, name, model_save_dir="ACGAN_model"):
        generator_path = os.path.join(model_save_dir, "{}-generator.pt".format(name))
        print(generator_path)
        assert os.path.isfile(generator_path), "Generator model file doesn't exist"
        self.generator.set_params(torch.load(generator_path))

    def generate_data(self, target_labels):
        """
        generate fake data corresponding to target labels
        :param target_labels: target labels for new fake data (list or numpy array list)
        :return: fake data tensor (image tensor)
        """
        # Sample noise
        z = Variable(self.FloatTensor(np.random.normal(0, 1, (len(target_labels), self.latent_dim)))).to(self.device)
        labels = np.array(target_labels)
        labels = Variable(self.LongTensor(labels)).to(self.device)
        self.generator.eval()
        return self.generator(z, labels)

    @staticmethod
    def results_to_dataset_tensor(generated_data, dataset_name):
        """
        transform GAN model output data to dataset tensor
        :param generated_data: GAN model generated data
        :param dataset_name: target dataset name
        :return:
        """
        transform_method_name = "gan_tensor_to_%s_data" % dataset_name
        transform_method_path = "utils.dataset_utils"
        import importlib
        mod = importlib.import_module(transform_method_path)
        transform_method = getattr(mod, transform_method_name)
        return transform_method(generated_data)

