import argparse

from helpers.script_functions import fix_seed, add_dict_to_namespace
from datasets.match_cycle import MatchCycleDatabase
from models.res_linear_model import ResLinearModel


def main(args):
    fix_seed(args.seed)
    database = MatchCycleDatabase(args.dataset_path, args.y_maximum, args.test_size, args.seed)
    args = add_dict_to_namespace(args, database.get_dims())
    model = ResLinearModel(args)
    model.fit(database.get_train_dataset(), database.get_val_dataset())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # global args
    parser.add_argument("--seed", type=int, default=0, help="random state.")
    parser.add_argument("--device", type=str, default="cuda", help="On which machine to run scripts.") 
    parser.add_argument("--project", type=str, default="match_cycle", help="wandb project name.")
    parser.add_argument("--run_name", type=str, default="debug", help="wandb run name.")
    parser.add_argument("--sweep", type=bool, default=False, help="If in sweeping mode.")
    # dataset args
    parser.add_argument("--dataset_path", type=str, default="inputs/match_cycle.txt", help="path for training data.")
    parser.add_argument("--y_maximum", type=str, default=40000, help="upper limit for filtering target value in dataset.")
    parser.add_argument("--test_size", type=float, default=0.1, help="ratio for spliting train, val & test datsaet.")
    # model args
    parser.add_argument("--chkpt_path", type=str, default="outputs/chkpts", help="path for storing and loading chekcpoints.")
    parser.add_argument("--d_model", type=int, default=128, help="dimension of model.")
    parser.add_argument("--num_layers", type=int, default=3, help="layers of model.")
    # training args
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate for training model.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum beta1 for Adam optimizer.")
    parser.add_argument("--adagrad", type=float, default=0.999, help="adagrad beta2 for Adam optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for training model.")
    parser.add_argument("--scheduler_step_size", type=int, default=99999, help="step size for learning rate step sheduler.") # default disabled
    parser.add_argument("--early_stopping", type=int, default=20, help="early stopping threshold.") # default disabled
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=100) # set to 0 for evaluation purpose
    parser.add_argument("--num_workers", type=int, default=8, help="num workers for torch dataloader.")
    # lambda for loss terms
    parser.add_argument("--lambda_mse", type=float, default=0, help="lambda for loss term mse.")
    parser.add_argument("--lambda_l1", type=float, default=0, help="lambda for loss term l1.")
    parser.add_argument("--lambda_ratio", type=float, default=0, help="lambda for loss term ratio.")
    args = parser.parse_args()
    main(args)