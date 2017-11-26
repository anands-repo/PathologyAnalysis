import argparse
import scipy.stats
import numpy as np

parser = argparse.ArgumentParser(description = "Create noisy data representing collective expression of a mixture of cell-types");

parser.add_argument("--composition", action="store", help="Comma-separated, mean cell type composition of sample (normalized)", dest="composition", required=True);
parser.add_argument("--scale", action="store", help="Scale factor for Dirichlet distribution", dest="scale", type=float, required=True);
parser.add_argument("--expression", action="store", help="File containing expression vector for each cell type (in npy format)", dest="expression", required=True);
parser.add_argument("--noise", action="store", help="Noise signal strength (positive)", type=float, dest="noise", required=True);
parser.add_argument("--num_samples", action="store", type=int, help="Number of samples to draw", required=True);
parser.add_argument("--output_prefix", action="store", help="Prefix of output file name", required=True);

args = parser.parse_args();

""" Meaning of scale in Dirichlet distribution: https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution """

alpha = [];

for i in args.composition.split(","):
    alpha.append(float(i) * args.scale);

# Generate 'num_samples' composition vectors
sample_compositions = scipy.stats.dirichlet.rvs(np.array(alpha), size=args.num_samples);

# Read the expression vectors - dimension = [len(alpha) x vector_length]
expression = np.load(args.expression);

# Generate 'num_samples' noise vectors
vector_length = expression.shape[1];

noise_vectors = scipy.stats.multivariate_normal.rvs(np.zeros(dtype=np.float32, shape=(vector_length,)), cov=args.noise);

# Generate samples
samples = np.matmul(sample_compositions, expression) + noise_vectors;

# Dump samples to file
np.save(args.output_prefix, samples);
np.savez(args.output_prefix + "_composition", a=np.array(alpha), b=args.scale);
