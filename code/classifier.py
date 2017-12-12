import numpy as np
import torch
import argparse
from read_gene_expression import gene_expression as reader
from copy import deepcopy

def classification_accuracy(labels, targets):
    labels      = np.argmax(labels, axis=1);
    num_correct = np.add.reduce(np.array(labels == targets, dtype=np.float32));
    return num_correct / labels.shape[0];

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train and test a pathology classifier");

    parser.add_argument("--pos_group", action="store", help="Comma-separated patient groups designated positive", dest="pos_group", required=True);
    parser.add_argument("--neg_group", action="store", help="Comma-separated patient groups designated negative", dest="neg_group", required=True);
    parser.add_argument("--expression", action="store", help="Gene expression file", dest="expression", required=True);
    parser.add_argument("--meta_data", action="store", help="Meta data file with label information", dest="meta_data", required=True);
    parser.add_argument("--marker_files", action="store", help="Comma-separated marker gene files", dest="marker_files", required=False);
    parser.add_argument("--train_test_val", action="store", help="Comma-separated train,test,validation split", dest="train_test_val", default="0.7,0.1,0.2");
    parser.add_argument("--num_folds", action="store", help="Number of folds of cross-validation to perform", dest="num_folds", type=int, default=1);
    parser.add_argument("--classifier_type", action="store", choices=["NN, DT, RF"], help="Type of classifier to use", dest="type", default="NN");
    parser.add_argument("--use_marker_genes", action="store_true", help="Turn on marker gene usage", dest="use_marker_genes", default=False);
    parser.add_argument("--train_marker_coefficients", action="store_true", help="Train marker coefficients during training", dest="train_coeffs", default=False);
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", help="Learning rate", default=1e-4, type=float);
    parser.add_argument("--batch_size", action="store", type=int, dest="batch_size", help="Batch size to use for training", default=10);
    parser.add_argument("--num_epochs", action="store", type=int, dest="num_epochs", help="Number of epochs to train", default=10);

    args = parser.parse_args();

    pos_group = args.pos_group.split(",");
    neg_group = args.neg_group.split(",");
    pos_ids   = [];
    neg_ids   = [];

    """ Open meta-data file and collect patient ids for positive and negative groups """
    with open(args.meta_data, 'r') as meta_data:
        for line in meta_data:
            items = line.split(",");

            if items[1] in pos_group:
                pos_ids.append(items[0]);

            if items[1] in neg_group:
                neg_ids.append(items[0]);

    print("Found %d patients for groups %s in positive group and %d patients for groups %s in negative group"%(len(pos_ids),args.pos_group,len(neg_ids),args.neg_group));

    """ Read marker genes for the different cell-types """
    marker_names = args.marker_files.split(",") if args.marker_files is not None else None;
    marker_lists = [];

    if marker_names is not None:
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

    """ Read expression file and obtain expressions for every patient """
    dataset = reader(args.expression);

    positive_vectors = dataset.gene_expression(pos_ids);
    positive_labels  = np.array([1] * positive_vectors.shape[0]);
    negative_vectors = dataset.gene_expression(neg_ids);
    negative_labels  = np.array([0] * negative_vectors.shape[0]);

    all_vectors      = np.concatenate([positive_vectors, negative_vectors]);
    all_labels       = np.concatenate([positive_labels, negative_labels]);
    indices          = np.arange(all_vectors.shape[0]);
    np.random.shuffle(indices);
    vectors          = all_vectors[indices];
    labels           = all_labels[indices];

    print("Read gene expression file %s", args.expression);

    """ Create a simple neural network """
    network = torch.nn.Sequential( \
                torch.nn.Linear(vectors[0].shape[0], 4096), \
                torch.nn.ReLU(), \
                torch.nn.Linear(4096, 2), \
    );

    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate);

    ftrain, ftest, fval = list(map(float, args.train_test_val.split(",")));
    ntrain              = int(all_vectors.shape[0] * ftrain);
    ntest               = int(all_vectors.shape[0] * ftest);
    nval                = all_vectors.shape[0] - (ntrain + ntest);

    for i in range(args.num_folds):
        train_vectors = vectors[:ntrain];
        train_labels  = labels[:ntrain];
        test_vectors  = vectors[ntrain:ntrain+ntest];
        test_labels   = labels[ntrain:ntrain+ntest];
        val_vectors   = vectors[ntrain+ntest:];
        val_labels    = labels[ntrain+ntest:];

        max_accuracy  = -1;
        best_model    = None;

        # Train-test iterations
        for j in range(args.num_epochs):

            def decide_num_batches(batch_size, vecs):
                num_batches = vecs.shape[0] // batch_size;

                if num_batches * batch_size < vecs.shape[0]:
                    num_batches += 1;

                return num_batches;

            num_batches = decide_num_batches(args.batch_size, train_vectors);

            # Train iterations
            for k in range(num_batches):
                batch_start = k * args.batch_size;
                batch_end   = min((k + 1) * args.batch_size, train_vectors.shape[0]);

                input_tensors = train_vectors[batch_start:batch_end];
                input_labels  = train_labels[batch_start:batch_end];
                predictions   = network(torch.autograd.Variable(torch.from_numpy(input_tensors).float()));
                targets       = torch.autograd.Variable(torch.from_numpy(input_labels));

                criterion     = torch.nn.CrossEntropyLoss();
                loss_function = criterion(predictions, targets);

                optimizer.zero_grad();
                loss_function.backward();
                optimizer.step();

            # Testing
            test_predictions = network(torch.autograd.Variable(torch.from_numpy(test_vectors)).float());
            test_accuracy    = classification_accuracy(test_predictions.data.numpy(), test_labels);

            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy;
                best_model   = deepcopy(network.state_dict());

            print("Completed epoch %d, obtained test accuracy %f"%(j,test_accuracy));

        # Validation iterations
        val_model = network.load_state_dict(best_model);

        test_predictions = network(torch.autograd.Variable(torch.from_numpy(test_vectors)));
        test_accuracy    = classification_accuracy(test_predictions.data.numpy(), test_labels);

        print("Fold %d sanity check: Validation model has test accuracy %f"%(i, test_accuracy));

        val_predictions = network(torch.autograd.Variable(torch.from_numpy(val_vectors)));
        val_accuracy    = classification_accuracy(val_predictions.data.numpy(), val_labels);

        print("Fold %d has validation accuracy %f"%(val_accuracy));

        # Circular right shift all vectors and labels by nval items
        vectors = np.roll(vectors, nval);
        labels  = np.roll(labels, nval);
