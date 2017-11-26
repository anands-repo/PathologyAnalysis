import numpy as np
import argparse
import scipy.stats
import numpy as np

parser = argparse.ArgumentParser(description = "Create random expression vectors");

parser.add_argument("--vector_length", action="store", help="Length of each expression vector", dest="length", required=True, type=int);
parser.add_argument("--num_components", action="store", help="Number of components", dest="num", required=True, type=int);
parser.add_argument("--output_prefix", action="store", help="Prefix of output file", dest="output_prefix", required=True);
parser.add_argument("--max_min", action="store", help="Comma separated duplet of minimum and maximum", dest="max_min", required=True);

args = parser.parse_args();

[maximum, minimum] = [int(x) for x in args.max_min.split(",")];

vectors = np.random.uniform(minimum, maximum, args.length*args.num).reshape((args.num, args.length));

np.save(args.output_prefix, vectors);
