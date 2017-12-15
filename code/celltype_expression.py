import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
from read_gene_expression import gene_expression as reader

def celltype_signals(marker_files, expression_reader, patient_ids):
    num_patients = len(patient_ids);
    marker_names = marker_files.split(",");
    marker_lists = [];

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

    marker_vectors = [];

    for marker_list in marker_lists:
        # Obtain expression vector over the marker support for each cell-type and average it
        expression_vectors_marker_specific = expression_reader.gene_specific_expression(patient_ids, marker_list);
        population_signal_for_marker       = np.add.reduce(expression_vectors_marker_specific, axis=1) / len(marker_list);

        marker_vectors.append(np.reshape(population_signal_for_marker,(num_patients,1)));

    celltype_compositions_unnormalized = np.concatenate(marker_vectors, axis=1);
    celltype_compositions              = celltype_compositions_unnormalized / np.expand_dims(np.add.reduce(celltype_compositions_unnormalized, axis=1), axis=1);

    return celltype_compositions;

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Find cell-type specific expression/coefficients through linear regression");

    parser.add_argument("--expression", action="store", help="Gene expression file", dest="expression", required=True);
    parser.add_argument("--group", action="store", help="Group of patients for which cell-type-specific expression is desired", dest="group", required=True);
    parser.add_argument("--meta_data", action="store", help="Meta data file with label information", dest="meta_data", required=True);
    parser.add_argument("--marker_files", action="store", help="Comma-separated marker gene files", dest="marker_files", required=True);
    parser.add_argument("--output_prefix", action="store", help="Prefix of output file to store cell-type-specific expression", dest="output_prefix", required=True);
    parser.add_argument("--patient_specific", action="store_true", help="Also determine cell-type specific expression for each patient", dest="patient_specific", default=False);
    
    args = parser.parse_args();

    dataset = reader(args.expression);

    """ Open meta-data file and collect patient ids for the given group """
    patient_ids = [];

    with open(args.meta_data, 'r') as meta_data:
        for line in meta_data:
            items = line.split(",");

            if items[1] in args.group:
                patient_ids.append(items[0]);

    num_patients = len(patient_ids);

    print("Obtained %d patients for group %s"%(num_patients, args.group));

    """ Estimate cell-type population signals from marker genes """
    celltype_compositions = celltype_signals(args.marker_files, dataset, patient_ids);
    
    """ Obtain complete gene expression data """
    expression = dataset.gene_expression(patient_ids);

    """ Determine celltype-specific expression coefficients by regression. cell-type composition is the source, expression is the target """
    weights = [];
    models  = [];

    regressor = LinearRegression(fit_intercept=False, n_jobs=4);

    regressor.fit(celltype_compositions, expression);

    """ Dump weights into file, weights to be reshaped as [celltype x genes] """
    weights = regressor.coef_.T; # coef_ is of shape [n_targets x n_features], targets = genes, features = celltype-composition

    print("Dumping weights of linear regressor of shape %s to %s"%(str(weights.shape), args.output_prefix + ".npy"));

    np.save(args.output_prefix, weights);
    np.save(args.output_prefix + "_composition", celltype_compositions);

    """ Separately fit a model to each patient if required by user """
    if args.patient_specific:
        patient_specific_expression  = np.split(expression, expression.shape[0]);
        patient_specific_composition = np.split(celltype_compositions, celltype_compositions.shape[0]);

        celltype_specific_expressions = [];

        for e, c in zip(patient_specific_expression, patient_specific_composition):
            regressor = LinearRegression(fit_intercept=False, n_jobs=4);

            # Input is composition, target is expression. This will give a [55k x 5] coefficient matrix. Reshape.
            regressor.fit(c, e);

            coeff = np.expand_dims(regressor.coef_.T.copy(), axis=1);

            celltype_specific_expressions.append(coeff);

        celltype_specific_expressions = np.concatenate(celltype_specific_expressions, axis=1);

        # 5 x #num_patients x 55k tensor
        np.save(args.output_prefix + "_patient_specific", celltype_specific_expressions);
