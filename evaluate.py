import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import Constants
from datasets import WebDataset, custom_collate_fn
from models import CoVA
from train import evaluate_model
from utils import cmdline_args_parser, print_and_log


def evaluate(
    model,
    test_loader,
    device,
    log_file,
    test_acc_imgwise_file,
    webpage_info=None,
    test_domains=None,
    test_acc_domainwise_file=None,
):
    """
    Evaluate trained model on test dataset and compute classwise, imgwise (webpagewise), and domainwise accuracies
    Return classwise_acc and macro_acc (per class average of accuracy of each domain) of type np.array [n_classes]
    """
    print(
        "Evaluating classwise, imgwise (webpagewise), and domainwise accuracies on test data..."
    )
    img_acc, class_acc_test = evaluate_model(
        model, test_loader, device, 1, "TEST", log_file
    )

    np.savetxt(
        test_acc_imgwise_file,
        img_acc,
        "%s,%.2f,%.2f,%.2f",
        ",",
        header="img_id,price_acc,title_acc,image_acc",
        comments="",
    )

    class_names = model.class_names
    if test_domains is None or webpage_info is None or test_acc_domainwise_file is None:
        macro_acc_test = np.zeros(len(class_names))
    else:  # compute macro accuracy
        with open(test_acc_domainwise_file, "w") as f:
            f.write(
                "Domain,N_examples,%s,%s,%s\n"
                % (class_names[1], class_names[2], class_names[3])
            )
            for domain in test_domains:
                domain_imgs = webpage_info[
                    np.isin(webpage_info[:, 1], domain), 0
                ].astype(np.int32)
                domain_class_acc = (
                    img_acc[np.isin(img_acc[:, 0], domain_imgs), 1:].mean(0) * 100
                )
                f.write(
                    "%s,%d,%.2f,%.2f,%.2f\n"
                    % (
                        domain,
                        len(domain_imgs),
                        domain_class_acc[0],
                        domain_class_acc[1],
                        domain_class_acc[2],
                    )
                )

        macro_acc_test = np.zeros(len(class_names))
        macro_acc_test[1:] = (
            np.loadtxt(test_acc_domainwise_file, delimiter=",", skiprows=1, dtype=str)[
                :, 2:
            ]
            .astype(np.float32)
            .mean(0)
        )
        for c in range(1, len(class_names)):  # class at index 0 is a 'Background' class
            print_and_log(
                "%s Macro Acc: %.2f%%" % (class_names[c], macro_acc_test[c]), log_file
            )

    return class_acc_test, macro_acc_test


if __name__ == "__main__":
    ########## CMDLINE ARGS - PROVIDE HYPERPARAMETERS OF TRAINED MODEL TO BE EVALUATED ##########
    parser = cmdline_args_parser()
    args = parser.parse_args()

    device = torch.device(
        "cuda:%d" % args.device if torch.cuda.is_available() else "cpu"
    )

    N_CLASSES = Constants.N_CLASSES
    CLASS_NAMES = Constants.CLASS_NAMES
    IMG_HEIGHT = Constants.IMG_HEIGHT
    DATA_DIR = Constants.DATA_DIR
    SPLIT_DIR = Constants.SPLIT_DIR
    OUTPUT_DIR = Constants.OUTPUT_DIR

    CV_FOLD = args.cv_fold
    FOLD_DIR = "%s/Fold-%d" % (SPLIT_DIR, CV_FOLD)
    if CV_FOLD == -1:
        FOLD_DIR = SPLIT_DIR  # use files from SPLIT_DIR

    test_img_ids = np.loadtxt("%s/test_imgs.txt" % FOLD_DIR, str)

    # for calculating domainwise and macro accuracy if below files are available (optional)
    webpage_info_file = "%s/webpage_info.csv" % FOLD_DIR
    webpage_info = None
    if os.path.isfile(webpage_info_file):
        webpage_info = np.loadtxt(
            webpage_info_file, str, delimiter=",", skiprows=1
        )  # (img_id, domain) values

    test_domains_file = "%s/test_domains.txt" % FOLD_DIR
    test_domains = None
    if os.path.isfile(test_domains_file):
        test_domains = np.loadtxt(test_domains_file, str)

    ########## HYPERPARAMETERS ##########
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    CONTEXT_SIZE = args.context_size
    use_context = CONTEXT_SIZE > 0
    HIDDEN_DIM = args.hidden_dim if use_context else 0
    ROI_OUTPUT = (args.roi, args.roi)
    BBOX_HIDDEN_DIM = args.bbox_hidden_dim
    USE_ADDITIONAL_FEAT = args.additional_feat
    WEIGHT_DECAY = args.weight_decay
    DROP_PROB = args.drop_prob
    SAMPLING_FRACTION = (
        args.sampling_fraction
        if (args.sampling_fraction >= 0 and args.sampling_fraction <= 1)
        else 1
    )

    params = (
        "lr-%.0e batch-%d cs-%d hd-%d roi-%d bbhd-%d af-%d wd-%.0e dp-%.1f sf-%.1f"
        % (
            LEARNING_RATE,
            BATCH_SIZE,
            CONTEXT_SIZE,
            HIDDEN_DIM,
            ROI_OUTPUT[0],
            BBOX_HIDDEN_DIM,
            USE_ADDITIONAL_FEAT,
            WEIGHT_DECAY,
            DROP_PROB,
            SAMPLING_FRACTION,
        )
    )
    results_dir = "%s/%s" % (OUTPUT_DIR, params)

    assert os.path.exists(
        results_dir
    ), "Model does not seem to have been trained (run main.py) with the hyperparameters you provided"

    ########## TEST DATA LOADER ##########
    test_dataset = WebDataset(
        DATA_DIR, test_img_ids, CONTEXT_SIZE, USE_ADDITIONAL_FEAT, sampling_fraction=1
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=5,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )
    n_additional_feat = test_dataset.n_additional_feat

    log_file = "Fold-%s test_acc_classwise.txt" % (
        CV_FOLD
    )  # classwise results saved here
    test_acc_imgwise_file = "Fold-%s test_acc_imgwise.csv" % (
        CV_FOLD
    )  # imgwise (webpagewise) results saved here
    test_acc_domainwise_file = "Fold-%s test_acc_domainwise.csv" % (
        CV_FOLD
    )  # domainwise results saved here
    model_save_file = "%s/Fold-%s saved_model.pth" % (results_dir, CV_FOLD)

    ########## RESTORE TRAINED MODEL ##########
    model = CoVA(
        ROI_OUTPUT,
        IMG_HEIGHT,
        N_CLASSES,
        use_context,
        HIDDEN_DIM,
        BBOX_HIDDEN_DIM,
        n_additional_feat,
        DROP_PROB,
        CLASS_NAMES,
    ).to(device)
    model.load_state_dict(torch.load(model_save_file, map_location=device))

    evaluate(
        model,
        test_loader,
        device,
        log_file,
        test_acc_imgwise_file,
        webpage_info,
        test_domains,
        test_acc_domainwise_file,
    )
