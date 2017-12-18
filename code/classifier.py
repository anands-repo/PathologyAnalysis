import numpy as np
import argparse
from read_gene_expression import gene_expression as reader
from celltype_expression import celltype_signals, celltype_expression
from sklearn.model_selection import KFold
from learners import train_nn, train_rf, train_logistic
from statsmodels.formula.api import ols
import pandas
import sklearn.metrics as metrics
from remove_group_effect import mask_group_effect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train and test a pathology classifier");

    parser.add_argument("--groups", action="store", help="Patient label-groups. Currently two label-groups are supported. First label-group is '+'. \
                       Each label-group is separated by a ';', and each label can be composed of multiple sub-groups", dest="groups", required=True);
    parser.add_argument("--expression", action="store", help="Gene expression file", dest="expression", required=True);
    parser.add_argument("--meta_data", action="store", help="Meta data file with label information", dest="meta_data", required=True);
    parser.add_argument("--marker_files", action="store", help="Comma-separated marker gene files", dest="marker_files", required=False);
    parser.add_argument("--train_test_val", action="store", help="Comma-separated train,test,validation split. \
                                For RF classifier, the test split is not meaningful", dest="train_test_val", default="0.7,0.1,0.2");
    parser.add_argument("--num_folds", action="store", help="Number of folds of cross-validation to perform", dest="num_folds", type=int, default=1);
    parser.add_argument("--classifier_type", action="store", choices=["NN", "RF", "LR"], help="Type of classifier to use", dest="type", default="NN");
    parser.add_argument("--remove_group_effects", action="store_true", help="Remove group effect (CellCODE)", dest="remove_group_effects", default=False);
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", help="Learning rate for NN", default=1e-4, type=float);
    parser.add_argument("--batch_size", action="store", type=int, dest="batch_size", help="Batch size to use for training", default=10);
    parser.add_argument("--num_epochs", action="store", type=int, dest="num_epochs", help="Number of epochs to train", default=30);
    parser.add_argument("--num_layers", action="store", type=int, dest="num_layers", help="If using a neural network, how many layers", default=2);
    parser.add_argument("--num_hidden", action="store", type=int, dest="num_hidden", help="If using a neural network, how many hidden units", default=256);
    parser.add_argument("--loss", action="store", dest="loss", choices=["CE","MSE"], help="Loss function for NN", default="CE");
    parser.add_argument("--stop_type", action="store", dest="stop_type", choices=["late_stop", "early_stop", "take_last"], \
                                help="Stop criterion for NN", default="early_stop");
    parser.add_argument("--dropout", action="store", dest="dropout", type=float, help="Dropout regularizer for NN", required=False);
    parser.add_argument("--bn", action="store_true", dest="bn", help="Turn on batch-normalization for NN", default=False);

    args = parser.parse_args();

    groups    = args.groups.split(";");

    pos_group = groups[0].split(",");
    neg_group = groups[1].split(",");
    pos_ids   = [];
    neg_ids   = [];

    num_pos_patients = 0;
    num_neg_patients = 0;

    """ Open meta-data file and collect patient ids for positive and negative groups """
    with open(args.meta_data, 'r') as meta_data:
        for line in meta_data:
            items = line.split(",");

            if items[1] in pos_group:
                pos_ids.append(items[0]);

            if items[1] in neg_group:
                neg_ids.append(items[0]);

    num_pos_patients = len(pos_ids);
    num_neg_patients = len(neg_ids);
    patient_ids      = pos_ids + neg_ids;

    print("Found %d patients for groups %s in positive group and %d patients for groups %s in negative group"% \
                                                    (num_pos_patients,pos_group,num_neg_patients,neg_group));

    """ Load expression data """
    dataset_ = reader(args.expression);

    """ If marker files are given, read marker lists """
    marker_lists = [];
    marker_names = None;
    marker_expressions = [];

    if args.marker_files is not None:
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

        for marker_list in marker_lists:
            marker_expressions.append(dataset_.gene_specific_expression(patient_ids, marker_list));

    positive_expression  = dataset_.gene_expression(pos_ids);
    negative_expression  = dataset_.gene_expression(neg_ids);
    positive_composition = None;
    negative_composition = None;
    positive_coeffs      = None;
    negative_coeffs      = None;

    if args.marker_files is not None:
        positive_composition = celltype_signals(marker_expressions, np.arange(num_pos_patients));
        negative_composition = celltype_signals(marker_expressions, num_pos_patients + np.arange(num_neg_patients));

    positive_labels  = np.array([1] * num_pos_patients);
    negative_labels  = np.array([0] * num_neg_patients);

    expressions      = np.concatenate([positive_expression, negative_expression]);
    labels           = np.concatenate([positive_labels, negative_labels]);
    compositions     = None;

    if args.marker_files is not None:
        compositions = np.concatenate([positive_composition, negative_composition]);

    """ Shuffle the data """
    indices = np.arange(labels.shape[0]);
    np.random.shuffle(indices);

    """ Divide indices into folds """
    [ftrain, ftest, fval] = list(map(float, args.train_test_val.split(",")));
    folder       = KFold(shuffle=True, n_splits=args.num_folds);
    fold_indices = [];

    for train_indices_, val_indices_ in folder.split(labels):
        ntrain = int(train_indices_.shape[0] * ftrain/(ftrain + ftest));

        train_indices = train_indices_[:ntrain];
        test_indices  = train_indices_[ntrain:];
        val_indices   = val_indices_;

        fold_indices.append((train_indices, test_indices, val_indices));
    """ ============> Data loading complete <============ """

    """ Different types of runs are possible """
    datasets = [];

    if (args.marker_files is not None):
        """ For each fold, generate residual dataset based on the training set for that fold """

        print("Preparing data for folds");

        for tr_, te_, va_ in fold_indices:
            """ If group effects should be removed, then compute marker selectors for each celltype """
            patient_ids_tr   = [patient_ids[t] for t in tr_];
            marker_selectors = [];

            if args.remove_group_effects:
                for marker_expression in marker_expressions:
                    # celltype_specific_marker_expression = dataset_.gene_specific_expression(patient_ids_tr, marker_list);
                    celltype_specific_marker_expression = marker_expression[tr_];
                    marker_selectors.append(mask_group_effect(celltype_specific_marker_expression, labels[tr_]));

                compositions = celltype_signals(marker_expressions, np.arange(len(patient_ids)), marker_selectors);

            x1 = compositions[tr_];
            x2 = x1 * np.expand_dims(labels[tr_], axis=1);
            y  = expressions[tr_];
            x  = np.concatenate([x1, x2], axis=1);

            w_, i_, regressor = celltype_expression(x, y, fit_intercept=True);

            def predict_residual(w, i, comp, exp):
                # w    := [5 x 20k]
                # i    := [20k]
                # comp := [n_patients x 5]
                # exp  := [n_patients x 20k]
                n_types    = comp.shape[1];
                w          = w[:n_types];

                # Compute estimated expression if data is explained by one half of w 
                est  = np.matmul(comp, w) + i;

                # Return residuals
                return exp - est; # np.add.reduce(est, axis = 0);

            dataset = predict_residual(w_, i_, compositions, expressions);
            datasets.append(dataset);

            print("Completed fold ...");
    else:
        """ Simply use the expression data directly """
        for tr_, te_, va_ in fold_indices:
            datasets.append(expressions);

    folds = [];

    for ((tr_, te_, va_), dataset) in zip(fold_indices, datasets):
        folds.append( \
            [ \
                (dataset[tr_], labels[tr_]), \
                (dataset[te_], labels[te_]), \
                (dataset[va_], labels[va_]) \
            ] \
        );

    acc = models = predictions = None;

    if args.type == "NN":
        acc, auc, models = \
            train_nn( \
                    args.num_layers, \
                    args.num_hidden, \
                    args.learning_rate, \
                    args.num_epochs, \
                    args.batch_size, \
                    folds, \
                    args.stop_type, \
                    args.loss, \
                    dropout=args.dropout, \
                    bn=args.bn \
            );

    elif args.type == "RF":
        acc, auc, models = train_rf(folds);
    else:
        acc, auc, models = train_logistic(folds);

    print("Cross validation accuracy is %f, auc is %f"%(acc, auc));
