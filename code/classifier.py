import numpy as np
import torch
import argparse
from read_gene_expression import gene_expression as reader
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from celltype_expression import celltype_signals
from sklearn.model_selection import KFold

def classification_accuracy(labels, targets, one_hot=True):
    if one_hot: labels = np.argmax(labels, axis=1);
    num_correct = np.add.reduce(np.array(labels == targets, dtype=np.float32));
    return num_correct / labels.shape[0];

class neural_network(torch.nn.Module):
    def __init__(self, dim, num_layers=2, hidden=256):
        super().__init__();

        self.layers = [];

        linear = torch.nn.Linear(dim, hidden if num_layers > 1 else 2);
        linear.weight.data.normal_(0.0,0.05);
        linear.bias.data.fill_(0.1);

        self.layers.append(linear);

        if num_layers > 1:
            for i in range(num_layers-2):
                linear = torch.nn.Linear(hidden, hidden);
                linear.weight.data.normal_(0.0,0.05);
                linear.bias.data.fill_(0.1);

                self.layers.append(linear);

            linear = torch.nn.Linear(hidden, 2);
            linear.weight.data.normal_(0.0,0.05);
            linear.bias.data.fill_(0.1);

            self.layers.append(linear);

        for i, linear in enumerate(self.layers):
            setattr(self, "linear"+ str(i), linear);

    def forward(self, tensor):
        o = tensor;

        for i, layer in enumerate(self.layers):
            o_ = layer(o);
            o  = o_ if i == len(self.layers)-1 else torch.nn.functional.relu(o_);

        return o;

def train_nn(num_layers, num_hidden, learning_rate, num_epochs, batch_size, folds):
    # Return the model picked in each fold, and the train, test, val predictions of the model in each case
    models       = [];
    predictions_ = [];

    average_validation = 0;

    for i, fold in enumerate(folds):
        train_vectors = fold[0][0];
        train_labels  = fold[0][1];
        test_vectors  = fold[1][0];
        test_labels   = fold[1][1];
        val_vectors   = fold[2][0];
        val_labels    = fold[2][1];

        print(train_vectors.shape);

        print(train_labels);

        max_accuracy  = -1;
        best_model    = None;

        network = neural_network(train_vectors.shape[1], num_layers=num_layers, hidden=num_hidden); 
        network.cuda();

        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate);

        # Train-test iterations
        for j in range(num_epochs):

            def decide_num_batches(batch_size, vecs):
                num_batches = vecs.shape[0] // batch_size;

                if num_batches * batch_size < vecs.shape[0]:
                    num_batches += 1;

                return num_batches;

            num_batches = decide_num_batches(batch_size, train_vectors);

            network.train(True);

            # Train iterations
            for k in range(num_batches):
                batch_start = k * args.batch_size;
                batch_end   = min((k + 1) * args.batch_size, train_vectors.shape[0]);

                input_tensors = train_vectors[batch_start:batch_end];
                input_labels  = train_labels[batch_start:batch_end];
                predictions   = network(torch.autograd.Variable(torch.from_numpy(input_tensors).float().cuda()));
                targets       = torch.autograd.Variable(torch.from_numpy(input_labels).cuda());

                criterion     = torch.nn.CrossEntropyLoss();
                loss_function = criterion(predictions, targets);

                optimizer.zero_grad();
                loss_function.backward();
                optimizer.step();

            network.train(False);

            # Train accuracy
            train_predictions = network(torch.autograd.Variable(torch.from_numpy(train_vectors)).float().cuda());
            train_accuracy    = classification_accuracy(train_predictions.cpu().data.numpy(), train_labels);

            # Testing
            test_predictions = network(torch.autograd.Variable(torch.from_numpy(test_vectors)).float().cuda());
            test_accuracy    = classification_accuracy(test_predictions.cpu().data.numpy(), test_labels);

            if args.late_stop:
                if test_accuracy >= max_accuracy:
                    max_accuracy = test_accuracy;
                    best_model   = deepcopy(network.state_dict());
            else:
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy;
                    best_model   = deepcopy(network.state_dict());

            print("Completed epoch %d, obtained train, test accuracy %f,%f"%(j,train_accuracy,test_accuracy));

        # Validation iterations
        val_model = network.load_state_dict(best_model);

        network.train(False);

        test_predictions = network(torch.autograd.Variable(torch.from_numpy(test_vectors).float().cuda()));
        test_accuracy    = classification_accuracy(test_predictions.cpu().data.numpy(), test_labels);

        print("Fold %d sanity check: Validation model has test accuracy %f"%(i, test_accuracy));

        val_predictions = network(torch.autograd.Variable(torch.from_numpy(val_vectors).float().cuda()));
        val_accuracy    = classification_accuracy(val_predictions.cpu().data.numpy(), val_labels);

        print("Fold %d has validation accuracy %f"%(i, val_accuracy));

        train_predictions = network(torch.autograd.Variable(torch.from_numpy(train_vectors).float().cuda()));

        predictions_.append((train_predictions.cpu().data.numpy(), test_predictions.cpu().data.numpy(), val_predictions.cpu().data.numpy()));

        average_validation += val_accuracy;

        models.append(best_model);

    return average_validation / len(folds), models, predictions_;

def train_rf(folds):
    models = [];

    average_validation = 0;
    models = [];

    for i, fold in enumerate(folds):
        train_vectors = fold[0][0];
        train_labels  = fold[0][1];
        test_vectors  = fold[1][0];
        test_labels   = fold[1][1];
        val_vectors   = fold[2][0];
        val_labels    = fold[2][1];

        max_accuracy  = -1;

        rf = RandomForestClassifier();
        rf.fit(train_vectors, train_labels);

        val_predictions = rf.predict(val_vectors);
        val_accuracy    = classification_accuracy(val_predictions, val_labels, one_hot=False); 

        print("Obtained validation accuracy %f after fold %d"%(val_accuracy, i));

        average_validation += val_accuracy;

        models.append(rf);

    return average_validation / len(folds), models;

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train and test a pathology classifier");

    parser.add_argument("--groups", action="store", help="Patient label-groups. Currently two label-groups are supported. First label-group is '+'. \
                       Each label-group is separated by a ';', and each label can be composed of multiple sub-groups", dest="groups", required=True);
    parser.add_argument("--expression", action="store", help="Gene expression file", dest="expression", required=False);
    parser.add_argument("--meta_data", action="store", help="Meta data file with label information", dest="meta_data", required=False);
    parser.add_argument("--marker_files", action="store", help="Comma-separated marker gene files", dest="marker_files", required=False);
    parser.add_argument("--type_specific_exp", action="store", help="Comma-separated cell-type specific expression coefficient files. \
                            Should follow the same order as the label-groups (one file for each group). If this is provided, \
                            marker_files should also be provided. If provided, classifiers will be trained based on cell-type specific expression. \
                            A logistic regressor will be trained on the predictions of the cell-type-specific classifiers to predict \
                            the final label class.", dest="type_specific_exp", required=False);
    parser.add_argument("--patient_celltype_exp", action="store", help="Comma-separated patient-specific and cell-type specific expression files. \
                            Should follow the same order as the label-groups (one file for each group). If this is provided, \
                            then marker files, type specific coefficients, and overall expression (and meta-data) \
                            shouldn't be provided. When provided, classifiers will be \
                            trained for each cell-type with expression for that cell-type as input and class label as output.", required=False);
    parser.add_argument("--train_test_val", action="store", help="Comma-separated train,test,validation split. \
                                For RF classifier, the val split is not meaningful", dest="train_test_val", default="0.7,0.1,0.2");
    parser.add_argument("--num_folds", action="store", help="Number of folds of cross-validation to perform", dest="num_folds", type=int, default=1);
    parser.add_argument("--classifier_type", action="store", choices=["NN", "RF"], help="Type of classifier to use", dest="type", default="NN");
    parser.add_argument("--train_marker_coefficients", action="store_true", help="Train marker coefficients during training", dest="train_coeffs", default=False);
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", help="Learning rate", default=1e-4, type=float);
    parser.add_argument("--batch_size", action="store", type=int, dest="batch_size", help="Batch size to use for training", default=10);
    parser.add_argument("--num_epochs", action="store", type=int, dest="num_epochs", help="Number of epochs to train", default=10);
    parser.add_argument("--num_layers", action="store", type=int, dest="num_layers", help="If using a neural network, how many layers", default=2);
    parser.add_argument("--num_hidden", action="store", type=int, dest="num_hidden", help="If using a neural network, how many hidden units", default=256);
    parser.add_argument("--late_stop", action="store_true", dest="late_stop", help="Late stopping for neural networks", default=False);

    args = parser.parse_args();

    if (args.type_specific_exp is not None):
        assert(args.marker_files is not None), "Marker files not provided, but cell-type specific expression coefficients provided";

    if (args.patient_celltype_exp is not None):
        assert((args.marker_files is None) and (args.type_specific_exp is None) and (args.expression is None) and (args.meta_data is None)),\
                                        "Patient-celltype expression is given; markers, coeffs expression and meta-data should not be provided";
    else:
        assert((args.expression is not None) and (args.meta_data is not None)), \
                                        "One of patient-celltype expression, or overall expression (and meta-data) is desired";

    groups    = args.groups.split(";");

    pos_group = groups[0].split(",");
    neg_group = groups[1].split(",");
    pos_ids   = [];
    neg_ids   = [];

    num_pos_patients = 0;
    num_neg_patients = 0;

    """ Open meta-data file and collect patient ids for positive and negative groups """
    if (args.meta_data is not None):
        with open(args.meta_data, 'r') as meta_data:
            for line in meta_data:
                items = line.split(",");

                if items[1] in pos_group:
                    pos_ids.append(items[0]);

                if items[1] in neg_group:
                    neg_ids.append(items[0]);

        num_pos_patients = len(pos_ids);
        num_neg_patients = len(neg_ids);

        print("Found %d patients for groups %s in positive group and %d patients for groups %s in negative group"% \
                                                        (num_pos_patients,pos_group,num_neg_patients,neg_group));

    """ Load expression data """
    if args.expression is not None:
        dataset = reader(args.expression);

        positive_expression  = dataset.gene_expression(pos_ids);
        negative_expression  = dataset.gene_expression(neg_ids);
        positive_composition = None;
        negative_composition = None;
        positive_coeffs      = None;
        negative_coeffs      = None;

        if args.marker_files is not None:
            positive_composition = celltype_signals(args.marker_files, dataset, pos_ids);
            negative_composition = celltype_signals(args.marker_files, dataset, neg_ids);

        positive_labels  = np.array([1] * num_pos_patients);
        negative_labels  = np.array([0] * num_neg_patients);

    if args.type_specific_exp is not None:
        coeff_filenames = args.type_specific_exp.split(",");
        positive_coeffs = np.load(coeff_filenames[0]);
        negative_coeffs = np.load(coeff_filenames[1]);

    if args.patient_celltype_exp is not None:
        patient_celltype_files        = args.patient_celltype_exp.split(",");
        positive_patient_celltype_exp = np.load(patient_celltype_files[0]);
        negative_patient_celltype_exp = np.load(patient_celltype_files[1]);

    """ ============> Data loading complete <============ """
    [ftrain, ftest, fval] = list(map(float, args.train_test_val.split(",")));

    assert(1 / fval == args.num_folds), "Validation size doesn't fit with fold size for cross-validation";

    """ Different types of runs are possible """
    if ((args.marker_files is not None) and (args.type_specific_exp is not None)) or (args.patient_celltype_exp is not None):
        # [5 x 55k] coefficients for both positive and negative examples
        # [# num patients x 5] cell-type-specific compositions for each patient
        # We need 5 sets of data for positive and negative datasets (10 sets in total)
        # Convert coefficients into [5 x #num_patients x 55k] array by tiling
        # Convert compositions to [5 x #num_patients x 1] array by transpoing and reshaping
        # Multiply the two arrays together
        def create_datasets(compositions, coefficients):
            num_patients = compositions.shape[0];
            compositions = np.expand_dims(compositions.T, axis=2);
            coefficients = np.tile(np.expand_dims(coefficients, axis=1), (1,num_patients,1));
            return compositions * coefficients;

        positive_dataset = None;
        negative_dataset = None;

        if args.patient_celltype_exp is not None:
            positive_dataset = positive_patient_celltype_exp;
            negative_dataset = negative_patient_celltype_exp;
        else:
            positive_dataset = create_datasets(positive_composition, positive_coeffs);
            negative_dataset = create_datasets(negative_composition, negative_coeffs);

        dataset          = np.concatenate([positive_dataset, negative_dataset], axis=1);
        labels           = np.array([1] * positive_dataset.shape[1] + [0] * negative_dataset.shape[1]);

        """ Shuffle them up! """
        indices          = np.arange(labels.shape[0]);
        np.random.shuffle(indices);
        dataset          = dataset[:,indices,:];
        labels           = labels[indices];

        """ Divide data into folds """
        folder = KFold(shuffle=True, n_splits=args.num_folds);

        fold_indices = [];

        for train_indices_, val_indices_ in folder.split(labels):
            ntrain = int(train_indices_.shape[0] * ftrain/(ftrain + ftest));

            train_indices = train_indices_[:ntrain];
            test_indices  = train_indices_[ntrain:];
            val_indices   = val_indices_;

            fold_indices.append((train_indices, test_indices, val_indices));

        celltype_specific_calls = [];
        
        for i, data in enumerate(np.split(dataset, dataset.shape[0])):
            data = np.squeeze(data, axis=0);

            folds = [];

            for tr_, te_, va_ in fold_indices:
                folds.append([(data[tr_], labels[tr_]), (data[te_], labels[te_]), (data[va_], labels[va_])]);

            acc, models, predictions = train_nn(args.num_layers, args.num_hidden, args.learning_rate, args.num_epochs, args.batch_size, folds);
            celltype_specific_calls.append(predictions);
            print("Cross validation accuracy is %f for cell-type-specific expression for celltype %d"%(acc, i));

        """ Build a logistic regressor on top of this, use neural netowrk training """
        folds = [];

        for i, (tr_, te_, va_) in enumerate(fold_indices):
            training_data = [];
            test_data     = [];
            val_data      = [];

            for pred in celltype_specific_calls:
                training_data.append(pred[i][0]);
                test_data.append(pred[i][1]);
                val_data.append(pred[i][2]);

            training_data = np.concatenate(training_data, axis=1);
            test_data     = np.concatenate(test_data, axis=1);
            val_data      = np.concatenate(val_data, axis=1);

            folds.append([(training_data, labels[tr_]), (test_data, labels[te_]), (val_data, labels[va_])]);

        acc, models, predictions = train_nn(1, 0, args.learning_rate, args.num_epochs, args.batch_size, folds);
        print("Cross validation accuracy is %f for ensemble learner on cell-type-specific classifications"%(acc));
