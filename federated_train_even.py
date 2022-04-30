import enum
import sys, os

ROOT_DIR = os.path.dirname(__file__)
YOLO_DIR = os.path.join(ROOT_DIR, "yolov5")
sys.path.append(ROOT_DIR)
sys.path.append(YOLO_DIR)

import torch
from train import main
from models.yolo import Model
from utils.torch_utils import select_device
from utils.general import intersect_dicts
from centralized_train import parse_opt

DATASET_PREFIX = "even_subdataset_"
TRAIN_FOLDER = os.path.join(ROOT_DIR, "training", "federated_even_distributed")
CLIENT_NUM = 5
ROUNDS = 2
AGGRE_FOLDER = os.path.join(TRAIN_FOLDER, "aggregated")

def getLastAggModel():
    agg_files = os.listdir(AGGRE_FOLDER)
    agg_files.sort()
    last_agg = os.path.join(AGGRE_FOLDER, agg_files[-1])
    return last_agg

def getLastWeightFile(client_i):
    training_dir = os.path.join(TRAIN_FOLDER, str(client_i))
    # To see how many exp folder are in this directory
    exp_folders = os.listdir(training_dir)
    exp_folders.sort()

    # Only retrieve the last folder
    exp_folder = exp_folders[-1]
    weight_file = os.path.join(training_dir, exp_folder, "weights", "last.pt")

    return weight_file    


def getModelStateDict(weights):
    device = select_device(0, batch_size=16)

    ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    yolo_model = Model(ckpt['model'].yaml, ch=3, nc=10).to(device)  # create
    exclude = ['anchor']
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, yolo_model.state_dict(), exclude=exclude)

    return csd

def FedAvg(round_num):
    # Read the state dict of every model
    dict_list = []
    for i in range(0, CLIENT_NUM):
        weight_path = getLastWeightFile(i)
        dict_list.append(getModelStateDict(weight_path))

    # Aggregate every weight
    dict_avg = {}
    for i, state_dict in enumerate(dict_list):
        for key_name in state_dict.keys():
            if i == 0:
                dict_avg[key_name] = state_dict[key_name]
            else:
                current_value = state_dict[key_name]
                last_avg = dict_avg[key_name]
                dict_avg[key_name] = i*last_avg/(i+1) + current_value/(i+1)

    # Save the tensor avg as 
    save_name = os.path.join(AGGRE_FOLDER, "aggregated_model_"+str(round_num)+".pt")
    torch.save(dict_avg, save_name)

if __name__ == "__main__":
    opt0 = parse_opt()
    opt1 = parse_opt()
    opt2 = parse_opt()
    opt3 = parse_opt()
    opt4 = parse_opt()

    opts = [opt0, opt1, opt2, opt3, opt4]
    for i, opt in enumerate(opts):
        opt.epochs = 1
        opt.data = os.path.join(ROOT_DIR, "dataset_yaml", DATASET_PREFIX + str(i) + ".yaml")
        opt.project = os.path.join(TRAIN_FOLDER, str(i))

    for opt in opts:
        print(opt.data)
        print(opt.project)

    first_training = True
    for i in range(ROUNDS):
        if not first_training:
            # If not the first time training, the aggregate model will be used
            init_model = getLastAggModel()
        else:
            init_model = os.path.join(YOLO_DIR, "yolov5s.pt")

        # Train the models once
        for opt in opts:
            # Set initial model
            opt.weights = init_model
            main(opt)

        # Aggregate the model
        FedAvg(i)

        print("<Round {} finished>".format(i))
