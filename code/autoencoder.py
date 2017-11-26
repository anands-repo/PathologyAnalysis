import torch
import numpy as np
import scipy.stats
import argparse
from functools import reduce

def weight_variable(shape):
    num_elems = reduce(lambda x, y: x * y, shape, 1);
    elems     = scipy.stats.truncnorm.rvs(-0.1,0.1,size=num_elems).reshape(shape);
    return torch.from_numpy(elems).float();

class autoencoder(torch.nn.Module):
    def __init__(self, num_components, vector_length):
        super().__init__();

        self.analysis  = torch.nn.Parameter(weight_variable((vector_length, num_components)));
        self.synthesis = torch.nn.Parameter(weight_variable((num_components, vector_length)));

    def forward(self, tensors):
        analysis  = torch.nn.functional.softmax(torch.matmul(tensors, self.analysis));
        synthesis = torch.matmul(analysis, self.synthesis);

        return analysis, synthesis;

parser = argparse.ArgumentParser(description = "Try to learn the composition of the given samples");

parser.add_argument("--samples", action="store", help="Files where samples are kept", dest="samples", required=True);
parser.add_argument("--num_components", action="store", type=int, help="The number of components", dest="num_components", required=True);
parser.add_argument("--num_iterations", action="store", type=int, help="Number of iterations to train", dest="num_iterations", required=True);
parser.add_argument("--batch_size", action="store", type=int, help="Batch size", dest="batch_size", required=True);
parser.add_argument("--output_prefix", action="store", help="Prefix of file in which to store component estimates", dest="output_prefix", required=True);
parser.add_argument("--learning_rate", action="store", help="Learning rate", dest="lr", default=1e-3, type=float);
parser.add_argument("--composition_weight", action="store", help="weight", dest="weight", default=1.0, type=float);
parser.add_argument("--init_exp_vector", action="store", help="Initialization for expression vectors", dest="init_exp_vector");

args = parser.parse_args();

samples = np.load(args.samples);

encoder = autoencoder(args.num_components, samples.shape[1]);

if args.init_exp_vector is not None:
    init_vectors = np.load(args.init_exp_vector);
    encoder.synthesis.data = torch.from_numpy(init_vectors).float();

num_samples = samples.shape[0];

num_batches = num_samples // args.batch_size;

if num_batches * args.batch_size < num_samples:
    num_batches += 1;

vector = torch.autograd.Variable(torch.from_numpy(np.array([1.0] * args.num_components)/args.num_components).float(), requires_grad=True);

all_parameters = [i for i in encoder.parameters()] + [vector];

optimizer = torch.optim.Adam(iter(all_parameters), lr=args.lr);

# Train the autoencoder
for i in range(args.num_iterations):
    total_loss = 0;

    for j in range(num_batches):
        batch_start = j * args.batch_size;
        batch_end   = min((j+1) * args.batch_size, num_samples);
        batch       = torch.autograd.Variable(torch.from_numpy(samples[batch_start:batch_end]).float(), requires_grad=False);
        batch_      = torch.stack([vector]*batch.size()[0]);
        zero        = torch.autograd.Variable(torch.zeros(batch.size()[0],args.num_components).float(), requires_grad=False);

        analysis, synthesis = encoder(batch);

        criterion1 = torch.nn.MSELoss();
        criterion2 = torch.nn.MSELoss();

        loss1     = criterion1(synthesis, batch);
        loss2     = criterion2(analysis-batch_, zero);
        loss      = loss1 + loss2 * args.weight;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        total_loss += loss;

    print("Completed iteration, total loss is %s"%(str(total_loss.data)));

# Obtain predictions for all the inputs
sample_torch        = torch.autograd.Variable(torch.from_numpy(samples).float(), requires_grad=False);
analysis, synthesis = encoder(sample_torch);

np.save(args.output_prefix + "_components", analysis.data.numpy());
np.save(args.output_prefix + "_predictions", synthesis.data.numpy());
