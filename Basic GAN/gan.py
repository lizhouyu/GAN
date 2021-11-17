import torch
import math
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger




# --------------------- Load the dataset --------------------- #
# define the format of images in the dataset
def mnist_data():
    compose = transforms.Compose( #chained a series of common image transforms
        [
            transforms.ToTensor(), #transfer the image to tensor type
            transforms.Normalize(mean=0.5, std=0.5) # normalize the pixel values in an image to 0 to 1 given the mean and std for each channel. In this example, the transform mean and std for (channel,height,width) should be (0.5, 0.5, 0.5) and (0.5, 0.5, 0.5)
        ]
    )
    out_dir = './dataset' #output the downloaded dataset to the assigned folder. Here the folder should be './dataset'
    return datasets.MNIST( # load the MNIST dataset
        root=out_dir, # output folder
        train=True, # whether to creat dataset from the training dataset
        transform=compose, # transform the images in the downloaded dataset following the assigned operations. Here it is the operations in compose
        download=True #whether download the data and store it in the local side
    )




# --------------------- Build the network --------------------- #
# Discriminator
# Given a image, a discriminator tries to output the probability that whehter this image is from a natural dataset
class DiscriminatorNet(torch.nn.Module): # create a discriminator class which inherets torch.nn.Module(Base class for all neural network modules)
    '''
    Python coding tip:
    in a class, each function will have a compulsory parameter called "self" as the first parameter
    when calling, we don't have to pass this "self" parameter into the function
    '''
    # define the constructor for the discriminator class
    # In the constructor, we have to give:
    #     1. some parameters of the network
    #     2. the structure of each layer of the neural network
    def __init__(self): # the "init" is surrounded by two underscores "__" (https://blog.csdn.net/qazwsxrx/article/details/107936711)
        super(DiscriminatorNet, self).__init__() # call the constructor of its parent class: first construct a neural network module framework
        n_features = 28 * 28 # the input size (how many features the input has)
        n_out = 1 #the output size: the output should be 1 number indicates the probability of sth. is in class Y

        '''
        we design out neurual network has 5 layers
        1 input layer + 3 hidden layer + 1 output layer
        input layer is already created by the transform operations

        Why we have 3 hidden layers? (Ref: https://youtu.be/aircAruvnKk)
        Assume we got an image with hand-written figure(that's what in the MNIST dataset), our network will try to understand it by analyzing its composition.
        A figure is composed by different patterns such as circle, line, slash, etc.
        And a pattern is also composed by different scratches like curve segment, line segment, etc. 
        This is a top-down view of how we recognize a figure.
        Thinking it from a buttom-up perspective. 
        The first layer of our neural network is trying to understand the scratches of a fiven figure.
        The second layer is trying to see which pattern did the recognized scratches form.
        The third layer is trying to assemble the patterns to see which figure is the given one.
        The output layer is trying to analyze the figure to tell whether this figure is generated or from a natural dataset
        '''
        # create the first hidden layer of the neural network
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024), # this layer has 1024 neurons which is fully connected to the input features
            nn.LeakyReLU(0.2), # this layer is activated with [leaky ReLU function](https://paperswithcode.com/method/leaky-relu) with the negtive slope assigned as 0.2
            nn.Dropout(0.3) # randomly pick 30% neurons in this layer to output 0 [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
        )

        # create the second hidden layer of the neural network
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512), # this layer has 512 neurons, which is also fully connected with the previous layer
            nn.LeakyReLU(0.2), # this layer is activated with [leaky ReLU function](https://paperswithcode.com/method/leaky-relu)
            nn.Dropout(0.3) # randomly pick 30% neurons in this layer to output 0 [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
        )

        # create the third hidden layer of the neural network
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256), # this layer has 512 neurons, and is fully connected
            nn.LeakyReLU(0.2), # this layer is activated with [leaky ReLU function](https://paperswithcode.com/method/leaky-relu)
            nn.Dropout(0.3) # randomly pick 30% neurons in this layer to output 0 [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
        )

        # create the output layer of the neural network
        self.out = nn.Sequential(
            nn.Linear(256, n_out), # the number of neurons in an output layer has been defined at the begining, this layer is also fully connected with the last hidden layer
            nn.Sigmoid() # the output layer is activated with a sigmoid function 
        )

    # define the forward function of the discriminator
    # the forward function represents the forward propagation stage
    # the forward function is the computation performed at each call
    def forward(self, x): # the forward function has a parameter x which indicated the input 
        x = self.hidden0(x) # pass the input into the first hidden layer
        x = self.hidden1(x) # pass the result from previous layer(1st hidden layer) into this layer(2nd hidden layer)
        x = self.hidden2(x) # pass the result from previous layer(2nd hidden layer) into this layer(3rd hidden layer)
        x = self.out(x) # pass the result from the last hiddent layer into the output layer
        return x # return the final x

# Generator
# A generator tries to generate an image from a set of random values
class GeneratorNet(torch.nn.Module):
    #define the constructor function for the generator
    def __init__(self):
        super(GeneratorNet, self).__init__() # also call the parent class's constructor first
        n_features = 100 # this generator is designed to accept 100 random values as the input
        n_out = 28*28 # the output will be a flattened 28*28 image

        '''
        the generator is also designed to have 5 layers
        1 input layer + 3 hidden layer + 1 output layer
        
        the reason of using 3 hidden layer is similar to the discriminator
        the first hidden layer generates scratches from given random values
        the second hidden layer composes the scratches into patterns
        the third hidden layer assemble the patterns into a figure
        '''
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256), # the first hidden layer has 256 neurons which are fully connected with the input features
            nn.LeakyReLU(0.2) # this layer is activated with a leaky relu function
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512), # the second hidden layer has 512 neurons and is fully connected with the previous layer
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),# the third hidden layer has 1024 neurons and is fully connected with the previous layer
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh() #activate the output using the Tanh function
        )

    def forward(self, x): # this forward function has an input of 100 random values
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

'''
Questions: Why using LeakyReLU, Sigmoid, and Tanh as the activation functions in the hidden layers, output layer of the discriminator, output layer of the generator?
'''




# --------------------- Sample some random values --------------------- #
'''
generate random vectors with random values following the Gaussian distribution

Args:
    number: the number of returned vectors
    size: the size of each random vector

Return:
    A 2D torch tensor with several random vectors
'''
def normal_distribution_random_value_vectors_generator(number, size):
    n = Variable(torch.randn(number, size))
    return n




# --------------------- Image-Vector Conversion --------------------- #
'''
convert 2d images to 1d vectors
Args:
    images: a 3d tensor includes multiple images with the same size
Return:
    a 2d tensor includes multiple 1-d vectors which are flattened input images 
'''
def images_to_vectors(images):
    return images.view(images.size(0), images.size(2)*images.size(3))

'''
convert 1d vectors to 2d images
Args:
    images: a 2d tensor includes multiple 1-d vectors with a perfect square number of values
Return:
    a 3d tensor includes multiple square images with the same size 
'''
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, int(math.sqrt(vectors.size(1))), int(math.sqrt(vectors.size(1))))





# --------------------- Generate target labels for the discriminator --------------------- #
'''
For the discriminator, we want to train it to distinguish the generated images from the real images

Hence, we label all the generated images as 0 and all the real images as 1
'''
def generate_labels(size, type):
    if type == 'fake':
        return Variable(torch.zeros(size, 1))
    if type == 'real':
        return Variable(torch.ones(size, 1))




# --------------------- Define the function for training the discriminator --------------------- #
'''
Function for training the discriminator
Args:
    d_optimizer: the optimizer used for training the discriminator
    loss: loss function to be used
    real_data: images from the dataset
    fake_data: images generated by the generator
    discriminator: discriminator to be trained
Returns:
    error: error computed by the loss function. Actually we will get 2 errors, one for training with real data(labels are 1) and the other for training with fake data(labels are 0). The final error is adding up these two errors
    prediction_real: the prediction given by the discriminator when feeding with the real data
    prediction_fake: the prediction given by the discriminator when feeding with the fake data
'''
def train_discriminator(d_optimizer, loss, real_data, fake_data, discriminator):
    # reset gradients
    d_optimizer.zero_grad()
    # train on real data
    prediction_real = discriminator(real_data) # feed the discriminator with real data and get its prediction result
    error_real = loss(prediction_real, generate_labels(real_data.size(0), 'real')) # calculate the error when training with the real data; BCE loss function takes 2 inputs: the prediction result and the actual ground truth, computing the difference between these two inputs based on certain metric;
    error_real.backward() # backward propagation with the calculated loss
    # train on fake data (the process is the same as the training process with the real data)
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, generate_labels(fake_data.size(0), 'fake'))
    error_fake.backward()
    # update weights and gradients using the optimizer
    d_optimizer.step()
    #return error and prediction results
    error = error_real + error_fake
    return error, prediction_real, prediction_fake




# --------------------- Define the function for training the generator --------------------- #
'''
Function for training the generator
Args:
    g_optimizer: the optimizer used for training the generator
    loss: loss function to be used
    random_value_vectors: the random value vectors used by the generator to generate fake images
    generator: the generator to be trained
    discriminator: the discriminator to distinguish images generated by the generator from the real images
Returns:
    error: error computed by the loss function
'''
def train_generator(g_optimizer, loss, random_value_vectors, generator, discriminator):
    # generate the fake data with the generator 
    fake_data = generator(random_value_vectors)
    # reset gradients
    g_optimizer.zero_grad()
    # feed the fake data to the discriminator to see how well the discriminator got fooled
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, generate_labels(fake_data.size(0), 'real'))
    error_fake.backward()
    # update the weights with gradients
    g_optimizer.step()
    # return the error made by the discriminator when feeding the fake data
    return error_fake




# --------------------- Define the function for training the GAN --------------------- #
'''
Function for training the GAN
Args:
    num_epochs: how many epochs are we going to train
    data_loader: the data loader for loading the dataset
    discriminator: the discriminator in the GAN which is ought to be trained
    generator: the generator in the GAN which is ought to be trained
    d_optimizer: the optimizer for the discriminator
    g_optimizer: the optimizer for the generator
    loss: the loss function used in training
    test_random_value_vectors: random value vectors for the generator to generate fake images for testing
    noise_size: size of the random value vectors for generating fake images
'''
def train(num_epochs, data_loader, discriminator, generator, d_optimizer, g_optimizer, loss, test_random_value_vectors, noise_size):
    #create logger
    logger = Logger(model_name='VGAN', data_name='MNIST')
    
    for epoch in range(num_epochs): # for each epoch
        for n_batch, (real_images,_) in enumerate(data_loader): #for each minibatch, fetch the batch id and the images(real_images) from the dataset
            # train the discriminator
            real_data = Variable(images_to_vectors(real_images))
            # generate fake data and detach. The detach operation makes sure here the gradients won't be calculatd for the generator
            random_value_vectors = normal_distribution_random_value_vectors_generator(real_images.size(0), noise_size) # generate random value vectors to seed the generator
            fake_data = generator(random_value_vectors).detach()
            # train the discriminator with real and fake data
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, loss, real_data, fake_data, discriminator)

            # train the generator
            random_value_vectors = normal_distribution_random_value_vectors_generator(real_images.size(0), noise_size)
            g_error = train_generator(g_optimizer, loss, random_value_vectors, generator, discriminator)

            # log batch error
            num_batches = len(data_loader)
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # display progress every few batches
            if(n_batch) % 100 == 0:
                test_images = vectors_to_images(generator(test_random_value_vectors))
                test_images = test_images.data
                num_test_samples = test_images.size(0)
                logger.log_images(
                    test_images, num_test_samples, 
                    epoch, n_batch, num_batches
                )

                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

if __name__ == '__main__':
    # load data to the local side
    data = mnist_data()

    # create data loader to load data to the program
    data_loader = torch.utils.data.DataLoader(
        data, # dataset: where to load the data
        batch_size=100, # batch size: divide all the images in a dataset into multiple batches, here we assign the size of each batch 
        shuffle=True # whether to shuffle the order of images
    )

    # initialize the discriminator
    discriminator = DiscriminatorNet()
    # initialize the generator
    generator = GeneratorNet()
        
    # get Optimizers for the discriminator and the generator
    '''
    here the Adam optimizer will be used

    Optimizers are fetched from the optim library

    Adam takes the network parameters and the learning rate as the parameter and return an optimizer function

    here the learning rate is assigned as 0.0002
    '''
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002) 

    # define the loss function
    '''
    here the [Binary Cross Entopy Loss (BCE Loss)](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) will be used

    The BCE Loss resembles the log-loss

    The mean loss will be taken for each minibatch
    '''
    loss = nn.BCELoss()

    #deine random vector's size
    noise_size = 100
    # generate test random value vectors(noise)
    num_test_samples = 16
    test_random_value_vectors = normal_distribution_random_value_vectors_generator(num_test_samples, noise_size)

    # train the GAN
    num_epochs = 200
    train(num_epochs, data_loader, discriminator, generator, d_optimizer, g_optimizer, loss, test_random_value_vectors, noise_size)