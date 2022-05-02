import enum
import sys, os
from copy import deepcopy

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

DATASET_PREFIX = "noniid_"
TRAIN_FOLDER = os.path.join(ROOT_DIR, "training", "federated_noniid_distributed")
CLIENT_NUM = 4
ROUNDS = 50
AGGRE_FOLDER = os.path.join(TRAIN_FOLDER, "aggregated")

def getLastAggModel():
    prefix = "aggregated_model_"
    agg_files = os.listdir(AGGRE_FOLDER)

    ids = []
    for agg_file in agg_files:
        agg_name = agg_file.split('.')[0]
        agg_id = int(agg_name.split('_')[-1])
        ids.append(agg_id)

    ids.sort()
    last_id = ids[-1]
    agg_file_name = "{}{}.pt".format(prefix, str(last_id))

    last_agg = os.path.join(AGGRE_FOLDER, agg_file_name)
    return last_agg

def getLastWeightFile(client_i):
    training_dir = os.path.join(TRAIN_FOLDER, str(client_i))
    # To see how many exp folder are in this directory
    exp_folders = os.listdir(training_dir)

    folder_ids = []
    for folder_name in exp_folders:
        if folder_name == "exp":
            folder_ids.append(0)
        else:
            folder_ids.append(int(folder_name[3:]))

    folder_ids.sort()

    if folder_ids[-1] == 0:
        exp_folder_name = "exp"
    else:
        exp_folder_name = "exp{}".format(str(folder_ids[-1]))

    # Only retrieve the last folder

    weight_file = os.path.join(training_dir, exp_folder_name, "weights", "last.pt")

    return weight_file     


def getModelStateDict(weights):
    device = select_device(0, batch_size=16)
    ckpt = deepcopy(torch.load(weights, map_location='cpu'))  # load checkpoint to CPU to avoid CUDA memory leak
    yolo_model = Model(ckpt['model'].yaml, ch=3, nc=20).to(device)  # create
    #exclude = ['anchor']
    #csd = ckpt['model'].float().state_dict()
    #csd = intersect_dicts(csd, yolo_model.state_dict(), exclude=exclude)

    return yolo_model, ckpt

def FedAvg(round_num):
    # Read the state dict of every model
    dict_list = []
    for i in range(0, CLIENT_NUM):
        weight_path = getLastWeightFile(i)
        print("Retrieving weight from: {}".format(weight_path))
        yolo_model, ckpt = getModelStateDict(weight_path)
        #exclude = ['anchor']
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, yolo_model.state_dict())
        dict_list.append(csd)

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
    if not os.path.isdir(AGGRE_FOLDER):
        os.mkdir(AGGRE_FOLDER)

    yolo_model.load_state_dict(dict_avg) 

    ckpt['model'] = yolo_model
    save_name = os.path.join(AGGRE_FOLDER, "aggregated_model_"+str(round_num)+".pt")
    torch.save(ckpt, save_name)

if __name__ == "__main__":
    opt0 = parse_opt()
    opt1 = parse_opt()
    opt2 = parse_opt()
    opt3 = parse_opt()

    opts = [opt0, opt1, opt2, opt3]
    for i, opt in enumerate(opts):
        opt.epochs = 1
        opt.data = os.path.join(ROOT_DIR, "data", DATASET_PREFIX + str(i) + ".yaml")
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
            init_model = os.path.join(YOLO_DIR, "yolov5n.pt")
            first_training = False

        # Train the models once
        for opt in opts:
            # Set initial model
            opt.weights = init_model
            main(opt)

        # Aggregate the model
        FedAvg(i)

        print("<Round {} finished>".format(i))
