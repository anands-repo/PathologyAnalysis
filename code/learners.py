import torch
import numpy as np
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from celltype_expression import celltype_signals
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

def classification_accuracy(labels, targets, one_hot=True):
    if one_hot:
        labels = np.argmax(labels, axis=1);
    else:
        labels = np.squeeze(np.where(labels > 0.5, np.ones(labels.shape), np.zeros(labels.shape)));

    num_correct = np.add.reduce(np.array(labels == targets, dtype=np.float32));

    return num_correct / labels.shape[0];

class neural_network(torch.nn.Module):
    def __init__(self, dim, num_layers=2, hidden=256, loss="CE", dropout=None, bn=False):
        super().__init__();

        num_outputs = 2 if loss == "CE" else 1;

        self.layers = [];

        if bn:
            self.batch_norm = torch.nn.BatchNorm1d(dim);
            self.layers.append(self.batch_norm);

        linear = torch.nn.Linear(dim, hidden if num_layers > 1 else num_outputs);
        linear.weight.data.normal_(0.0,0.05);
        linear.bias.data.fill_(0.1);

        self.layers.append(linear);

        if num_layers > 1:
            for i in range(num_layers-2):
                linear = torch.nn.Linear(hidden, hidden);
                linear.weight.data.normal_(0.0,0.05);
                linear.bias.data.fill_(0.1);

                self.layers.append(linear);

            if dropout is not None:
                self.dropout = torch.nn.Dropout(p=dropout);

            linear = torch.nn.Linear(hidden, num_outputs);
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

            if (i == len(self.layers)-2) and hasattr(self, 'dropout'):
                o = self.dropout(o);

        return o;

def train_nn(num_layers, num_hidden, learning_rate, num_epochs, batch_size, folds, stop_type="early_stop", loss="CE", dropout=None, bn=False):
    # Return the model picked in each fold, and the train, test, val predictions of the model in each case
    models       = [];
    predictions_ = [];

    average_validation = 0;
    average_auc        = 0;

    for i, fold in enumerate(folds):
        train_vectors = fold[0][0];
        train_labels  = fold[0][1];
        test_vectors  = fold[1][0];
        test_labels   = fold[1][1];
        val_vectors   = fold[2][0];
        val_labels    = fold[2][1];

        max_accuracy  = -1;
        best_model    = None;

        network = neural_network(train_vectors.shape[1], num_layers=num_layers, hidden=num_hidden, loss=loss, dropout=dropout, bn=bn); 
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
                batch_start = k * batch_size;
                batch_end   = min((k + 1) * batch_size, train_vectors.shape[0]);

                input_tensors = train_vectors[batch_start:batch_end];
                input_labels  = train_labels[batch_start:batch_end];
                predictions   = network(torch.autograd.Variable(torch.from_numpy(input_tensors).float().cuda()));
                targets       = torch.autograd.Variable(torch.from_numpy(input_labels).cuda());

                criterion     = torch.nn.CrossEntropyLoss();

                if loss == "MSE":
                    criterion     = torch.nn.MSELoss();
                    targets       = targets.float();

                loss_function = criterion(predictions, targets);

                optimizer.zero_grad();
                loss_function.backward();
                optimizer.step();

            network.train(False);

            # Train accuracy
            train_predictions = network(torch.autograd.Variable(torch.from_numpy(train_vectors)).float().cuda());
            train_accuracy    = classification_accuracy(train_predictions.cpu().data.numpy(), train_labels, one_hot = (loss == "CE"));

            # Testing
            test_predictions = network(torch.autograd.Variable(torch.from_numpy(test_vectors)).float().cuda());
            test_accuracy    = classification_accuracy(test_predictions.cpu().data.numpy(), test_labels, one_hot = (loss == "CE"));

            if stop_type == "late_stop":
                if test_accuracy >= max_accuracy:
                    max_accuracy = test_accuracy;
                    best_model   = deepcopy(network.state_dict());
            elif stop_type == "early_stop":
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy;
                    best_model   = deepcopy(network.state_dict());
            else:
                max_accuracy = test_accuracy;
                best_model   = deepcopy(network.state_dict());

            print("Completed epoch %d, obtained train, test accuracy %f,%f"%(j,train_accuracy,test_accuracy));

        # Validation iterations
        val_model = network.load_state_dict(best_model);

        network.train(False);

        test_predictions = network(torch.autograd.Variable(torch.from_numpy(test_vectors).float().cuda()));
        test_accuracy    = classification_accuracy(test_predictions.cpu().data.numpy(), test_labels, one_hot = (loss == "CE"));

        print("Fold %d sanity check: Validation model has test accuracy %f"%(i, test_accuracy));

        val_predictions = network(torch.autograd.Variable(torch.from_numpy(val_vectors).float().cuda()));
        val_accuracy    = classification_accuracy(val_predictions.cpu().data.numpy(), val_labels, one_hot = (loss == "CE"));

        print("Fold %d has validation accuracy %f"%(i, val_accuracy));

        average_validation += val_accuracy;

        models.append(best_model);

        val_predictions = val_predictions.cpu().data.numpy()[:,1];

        # Find the ROC
        fpr, tpr, thresh_ = metrics.roc_curve(val_labels, val_predictions);

        # Find the AUC
        auc = metrics.auc(fpr, tpr);

        average_auc += auc;

    return average_validation / len(folds), average_auc / len(folds), models;

def train_logistic(folds):
    models = [];

    average_validation = 0;
    average_auc        = 0;
    models = [];

    predictions = [];

    for i, fold in enumerate(folds):
        train_vectors = fold[0][0];
        train_labels  = fold[0][1];
        test_vectors  = fold[1][0];
        test_labels   = fold[1][1];
        val_vectors   = fold[2][0];
        val_labels    = fold[2][1];

        max_accuracy  = -1;

        lr = LogisticRegression();
        lr.fit(train_vectors, train_labels);

        val_predictions = lr.predict(val_vectors);
        val_accuracy    = classification_accuracy(val_predictions, val_labels, one_hot=False); 

        train_predictions = lr.predict(train_vectors);

        print("Obtained validation accuracy %f after fold %d"%(val_accuracy, i));

        average_validation += val_accuracy;

        models.append(lr);

        # model predictions to compute auc
        prob = lr.predict_proba(val_vectors)[:,1];
        
        # Find the ROC
        fpr, tpr, thresh_ = metrics.roc_curve(val_labels, prob);

        # Find the AUC
        auc = metrics.auc(fpr, tpr);

        average_auc += auc;

    return average_validation / len(folds), average_auc / len(folds), models; #, predictions;

def train_rf(folds):
    models = [];

    average_validation = 0;
    average_auc        = 0;
    models = [];

    predictions = [];

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

        train_predictions = rf.predict(train_vectors);

        print("Obtained validation accuracy %f after fold %d"%(val_accuracy, i));

        average_validation += val_accuracy;

        models.append(rf);

        # model predictions to compute auc
        prob = rf.predict_proba(val_vectors)[:,1];
        
        # Find the ROC
        fpr, tpr, thresh_ = metrics.roc_curve(val_labels, prob);

        # Find the AUC
        auc = metrics.auc(fpr, tpr);

        average_auc += auc;

    return average_validation / len(folds), average_auc / len(folds), models; #, predictions;
