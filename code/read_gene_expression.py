import numpy as np

""" Class for reading gene expression data """
class gene_expression:
    def __init__(self, filename):
        lines = [];

        with open(filename, 'r') as fhandle:
            for line in fhandle:
                lines.append(line.rstrip());

        legend = lines[0];
        data   = lines[1:];

        self.genes = legend.split()[1:];

        self.data  = {};

        for line in data:
            items = line.split();
            key   = items[0];
            value = list(map(float, items[1:]));

            self.data[key] = np.array(value);

    """ Get expression data for a set of patients for a set of genes """
    def gene_specific_expression(self, patient_ids, gene_ids):
        gene_indices = np.array([i for i, g in enumerate(self.genes) if g in gene_ids]);
        expr_tensor  = [];

        for pid in patient_ids:
            expr_tensor.append(self.data[pid][gene_indices]);

        return np.array(expr_tensor);

    """ Get expression data for a set of patients """
    def gene_expression(self, patient_ids):
        expr_tensor = np.array([self.data[pid] for pid in patient_ids]);
        return expr_tensor;
