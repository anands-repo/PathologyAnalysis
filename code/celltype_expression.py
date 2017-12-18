import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
from read_gene_expression import gene_expression as reader

# def celltype_signals(marker_lists, expression_reader, patient_ids, marker_selectors = None):
def celltype_signals(marker_expressions, patient_indices, marker_selectors = None):
    marker_vectors = [];
    num_patients   = len(patient_indices);

    for i, marker_expression_ in enumerate(marker_expressions):
        marker_expression = marker_expression_[patient_indices];
        divisor = marker_expression.shape[1];

        if marker_selectors is not None:
            divisor = np.sum(marker_selectors[i]);

        # Obtain expression vector over the marker support for each cell-type and average it
        expression_vectors_marker_specific = marker_expression;

        if marker_selectors is not None:
            expression_vectors_marker_specific = np.matmul(expression_vectors_marker_specific, marker_selectors[i]);

        population_signal_for_marker       = np.add.reduce(expression_vectors_marker_specific, axis=1) / divisor;

        marker_vectors.append(np.reshape(population_signal_for_marker,(num_patients,1)));

    celltype_compositions_unnormalized = np.concatenate(marker_vectors, axis=1);
    celltype_compositions              = celltype_compositions_unnormalized / np.expand_dims(np.add.reduce(celltype_compositions_unnormalized, axis=1), axis=1);

    return celltype_compositions;

def celltype_expression(celltype_compositions, expression, fit_intercept=True):
    regressor = LinearRegression(n_jobs=4, fit_intercept=fit_intercept);

    regressor.fit(celltype_compositions, expression);

    weights   = regressor.coef_.T;     # [n_celltypes, n_genes]
    
    if fit_intercept:
        intercept = regressor.intercept_;  # [n_genes]
    else:
        intercept = np.zeros(weights.shape[1]);

    return weights, intercept, regressor;

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Find cell-type specific expression/coefficients through linear regression");

    parser.add_argument("--expression", action="store", help="Gene expression file", dest="expression", required=True);
    parser.add_argument("--group", action="store", help="Group of patients for which cell-type-specific expression is desired", dest="group", required=True);
    parser.add_argument("--meta_data", action="store", help="Meta data file with label information", dest="meta_data", required=True);
    parser.add_argument("--marker_files", action="store", help="Comma-separated marker gene files", dest="marker_files", required=True);
    parser.add_argument("--output_prefix", action="store", help="Prefix of output file to store cell-type-specific expression", dest="output_prefix", required=True);
    
    args = parser.parse_args();

    dataset = reader(args.expression);

    """ Open meta-data file and collect patient ids for the given group """
    patient_ids = [];

    with open(args.meta_data, 'r') as meta_data:
        for line in meta_data:
            items = line.split(",");

            if items[1] in args.group:
                patient_ids.append(items[0]);

    """ If marker files are given, read marker lists """
    marker_lists = [];
    marker_names = args.marker_files.split(",");

    for marker_name in marker_names:
        with open(marker_name, 'r') as fhandle:
            lines       = [];
            marker_list = [];

            for line in fhandle:
                lines.append(line);

            for line in lines[1:]:
                items = line.split();
                marker_list.append(items[0][1:-1]);

            marker_lists.append(marker_list);

    num_patients = len(patient_ids);

    print("Obtained %d patients for group %s"%(num_patients, args.group));

    """ Estimate cell-type population signals from marker genes """
    celltype_compositions = celltype_signals(marker_lists, dataset, patient_ids);
    
    """ Obtain complete gene expression data """
    expression = dataset.gene_expression(patient_ids);

    """ Determine celltype-specific expression coefficients by regression. cell-type composition is the source, expression is the target """
    weights, intercept, r_ = celltype_expression(celltype_compositions, expression);

    print("Dumping weights of linear regressor of shape %s to %s"%(str(weights.shape), args.output_prefix + ".npy"));

    np.savez(args.output_prefix, **{'weights':weights, 'intercept':intercept});
    np.save(args.output_prefix + "_composition", celltype_compositions);
