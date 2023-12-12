import logging
import argparse
import os
import numpy as np



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_3class(gt_data_file, pred_data_file):
    model_preds = np.load(pred_data_file)
    labels = np.load(gt_data_file)

    correct = np.count_nonzero(np.equal(model_preds, labels))

    logger.info(f'Accuracy: {100. * correct / len(labels):.2f}')


def check_regression(gt_data_file, pred_data_file, threshold):
    model_values = np.load(pred_data_file).squeeze()
    label_values = np.load(gt_data_file)

    model_preds_2class = np.greater_equal(model_values, 0)
    labels_2class = np.greater_equal(label_values, 0)
    correct_2class = np.count_nonzero(np.equal(model_preds_2class, labels_2class))
    
    model_preds = np.zeros_like(model_values)
    labels = np.zeros_like(label_values)
    model_preds[model_values > threshold] = 0
    model_preds[model_values < -threshold] = 2
    model_preds[(model_values >= -threshold) & (model_values <= threshold)] = 1
    labels[label_values > threshold] = 0
    labels[label_values < -threshold] = 2
    labels[(label_values >= -threshold) & (label_values <= threshold)] = 1
    correct_3class = np.count_nonzero(np.equal(model_preds, labels))

    MSE_val = np.mean((model_values - label_values) ** 2)
    
    logger.info(f'Accuracy_3class: {100. * correct_3class / len(labels):.2f}')
    logger.info(f'Accuracy_2class: {100. * correct_2class / len(labels):.2f}')
    logger.info(f'MSE={MSE_val:.3f}')


def main():
    # config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--TASK",
        default="regression",
        type=str,
        required=True,
        help="choose from ['regression', 'classification']",
    )
    parser.add_argument(
        "--data_folder",
        default='project/models/regression/results',
        type=str,
        help="Path to the stored ground-truth labels file",
    )
    parser.add_argument(
        "--threshold", default=0.03, type=float, help="The threshold for price percentage classification if regression is used."
    )

    args = parser.parse_args()

    gt_data_file = os.path.join(args.data_folder, 'labels.npy')
    pred_data_file = os.path.join(args.data_folder, 'model_preds.npy')

    if args.TASK == 'regression':
        check_regression(gt_data_file, pred_data_file, args.threshold)
    else:
        check_3class(gt_data_file, pred_data_file)


if __name__ == "__main__":
    main()