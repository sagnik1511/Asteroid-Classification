import os
import sklearn.metrics as metrics


def make_training_report(model, x_train, y_train, x_val, y_val, x_test, y_test, directory):
    filename = f"{len(os.listdir(directory)) + 1}.txt"
    path = os.path.join(directory, filename)
    with open(path, "wt") as file:
        # loading prediction of the training and validation data
        train_prediction = model.predict(x_train)
        val_prediction = model.predict(x_val)
        test_prediction = model.predict(x_test)

        # Accuracy Score
        train_acc = metrics.accuracy_score(y_train, train_prediction) * 100.0
        val_acc = metrics.accuracy_score(y_val, val_prediction) * 100.0
        test_acc = metrics.accuracy_score(y_test, test_prediction) * 100.0
        print(f"Training Accuracy : {'%.6f' % train_acc}  ||  Validation Accuracy : {'%.6f' % val_acc}  ||  "
              f"Testing Accuracy : {'%.6f' % test_acc}", file=file)

        # Precision Score
        train_p = metrics.precision_score(y_train, train_prediction, average="weighted") * 100.0
        val_p = metrics.precision_score(y_val, val_prediction, average="weighted") * 100.0
        test_p = metrics.precision_score(y_test, test_prediction, average="weighted") * 100.0
        print(f"Training Precision : {'%.6f' % train_p}  ||  Validation Precision : {'%.6f' % val_p}  ||  "
              f"Testing Precision : {'%.6f'% test_p}", file=file)

        # Recall Score
        train_r = metrics.recall_score(y_train, train_prediction, average="weighted") * 100.0
        val_r = metrics.recall_score(y_val, val_prediction, average="weighted") * 100.0
        test_r = metrics.recall_score(y_test, test_prediction, average="weighted") * 100.0
        print(f"Training Recall : {'%.6f' % train_r}  ||  Validation Recall : {'%.6f' % val_r}  ||  "
              f"Testing Recall : {'%.6f'% test_r}", file=file)

        # F1 Score
        train_f1 = metrics.f1_score(y_train, train_prediction, average="weighted") * 100.0
        val_f1 = metrics.f1_score(y_val, val_prediction, average="weighted") * 100.0
        test_f1 = metrics.f1_score(y_test, test_prediction, average="weighted") * 100.0
        print(f"Training F1 Score : {'%.6f' % train_f1}  ||  Validation F1 Score : {'%.6f' % val_f1}  ||  "
              f"Testing F1 Score : {'%.6f' % test_f1}", file=file)

        file.close()
        print(f"report saved at {path}")
