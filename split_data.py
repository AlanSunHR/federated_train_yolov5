import sys, os
from shutil import copyfile, copytree

ROOT_DIR = os.path.dirname(__file__)

ORIG_DATA_FOLDER = os.path.join(ROOT_DIR, "datasets", "VisDrone", "VisDrone2019-DET-train")
ORIG_ANNOT_FOLDER = os.path.join(ORIG_DATA_FOLDER, "annotations")
ORIG_IMAGE_FOLDER = os.path.join(ORIG_DATA_FOLDER, "images")
ORIG_LABEL_FOLDER = os.path.join(ORIG_DATA_FOLDER, "labels")

OUTPUT_ROOT = os.path.join(ROOT_DIR, "datasets")

def parseDataName(data_name):
    '''
        Given the file name 0000002_00005_d_0000014.txt,
        return: {'cata': '0000002'; 'name': 0000002_00005_d_0000014.txt}
    '''
    cata_name = data_name.split("_")[0]
    without_suffix = data_name.split(".")[0]
    
    return {'cata': cata_name, 'name': without_suffix}

def evenSplit(orig_folders, output_root, data_prefix="client", split_num=5):
    '''
        Evenly split the data into several sub-dataset
    '''
    orig_ann = orig_folders[0]
    orig_img = orig_folders[1]
    orig_lab = orig_folders[2]

    ann_suffix = '.txt'
    img_suffix = '.jpg'
    lab_suffix = '.txt'

    data_names = os.listdir(orig_ann)
    # sort the names
    data_names.sort()
    # covert data names to a friendlier expression
    for i, data_name in enumerate(data_names):
        new_name = parseDataName(data_name)
        data_names[i] = new_name
    
    for i in range(0, split_num):
        parent_folder_name = "{}_{}".format(data_prefix, str(i))
        # make directories for each client
        parent_folder_dir = os.path.join(output_root, parent_folder_name)
        if os.path.isdir(parent_folder_dir):
            print("Folder {} already exist! ".format(parent_folder_dir))
            exit(0)
        # If not exist, mkdir
        os.mkdir(parent_folder_dir)
        # Inside the folder, create folders
        dst_ann_folder = os.path.join(parent_folder_dir, "annotations")
        dst_img_folder = os.path.join(parent_folder_dir, "images")
        dst_lab_folder = os.path.join(parent_folder_dir, "labels")
        os.mkdir(dst_ann_folder)
        os.mkdir(dst_img_folder)
        os.mkdir(dst_lab_folder)

    # Begin to copy images from orig to dst folders
    current_client = 0
    for data_i, data_name in enumerate(data_names):
        # Still in the same catagory
        if current_client == split_num - 1:
            # Reaches the maximum client id
            current_client = 0        
        else:
            # Otherwise just increment it by 1
            current_client += 1

        # Get folder information
        parent_folder_name = "{}_{}".format(data_prefix, str(current_client))
        parent_folder_dir = os.path.join(output_root, parent_folder_name)
        dst_ann_folder = os.path.join(parent_folder_dir, "annotations")
        dst_img_folder = os.path.join(parent_folder_dir, "images")
        dst_lab_folder = os.path.join(parent_folder_dir, "labels")
        
        current_file_name = data_name['name']
        # Copy the annotation
        orig_path = os.path.join(orig_ann, current_file_name + ann_suffix)
        dst_path = os.path.join(dst_ann_folder, current_file_name + ann_suffix)
        copyfile(orig_path, dst_path)
        # Copy the image
        orig_path = os.path.join(orig_img, current_file_name + img_suffix)
        dst_path = os.path.join(dst_img_folder, current_file_name + img_suffix)
        copyfile(orig_path, dst_path)        
        # Copy the label
        orig_path = os.path.join(orig_lab, current_file_name + lab_suffix)
        dst_path = os.path.join(dst_lab_folder, current_file_name + lab_suffix)
        copyfile(orig_path, dst_path)  

        print("Copy image {} to client {}".format(current_file_name, current_client))      
        
def noniidSplit(orig_folders, output_root, data_prefix="client", split_num=5, split_ratios=[0.1, 0.1, 0.2, 0.3, 0.3]):
    '''
        Split the dataset block by block;

        Given the split_num (the number of subdataset you want), and split ratio
        (for each subdataset, the ratio of it compared with the whole dataset), 
        split the data block by block.
    '''     
    assert len(split_ratios) == split_num
    assert sum(split_ratios) == 1

    #  
    orig_ann = orig_folders[0]
    orig_img = orig_folders[1]
    orig_lab = orig_folders[2]

    ann_suffix = '.txt'
    img_suffix = '.jpg'
    lab_suffix = '.txt'

    data_names = os.listdir(orig_ann)
    # sort the names
    data_names.sort()
    # covert data names to a friendlier expression
    for i, data_name in enumerate(data_names):
        new_name = parseDataName(data_name)
        data_names[i] = new_name
    
    # Calculate the splitting indexes
    split_indices = [] # In form of [[start1, end1], [start2, end2], ...]
    for client_i, split_ratio in enumerate(split_ratios):
        if client_i == 0:
            start_index = 0
        else:
            start_index = split_indices[client_i -1][-1] + 1
        
        dataset_size = len(data_names)
        sub_size = int(dataset_size * split_ratio)


        if client_i == split_num - 1:
            end_index = dataset_size - 1
        else:
            end_index = sub_size + start_index

        split_indices.append([start_index, end_index])

    for i in range(0, split_num):
        parent_folder_name = "{}_{}".format(data_prefix, str(i))
        # make directories for each client
        parent_folder_dir = os.path.join(output_root, parent_folder_name)
        if os.path.isdir(parent_folder_dir):
            print("Folder {} already exist! ".format(parent_folder_dir))
            exit(0)
        # If not exist, mkdir
        os.mkdir(parent_folder_dir)
        # Inside the folder, create folders
        dst_ann_folder = os.path.join(parent_folder_dir, "annotations")
        dst_img_folder = os.path.join(parent_folder_dir, "images")
        dst_lab_folder = os.path.join(parent_folder_dir, "labels")
        os.mkdir(dst_ann_folder)
        os.mkdir(dst_img_folder)
        os.mkdir(dst_lab_folder)

    # According to the split index, copy the files
    for i, split_index in enumerate(split_indices):
        parent_folder_name = "{}_{}".format(data_prefix, str(i))
        # make directories for each client
        parent_folder_dir = os.path.join(output_root, parent_folder_name)        
        dst_ann_folder = os.path.join(parent_folder_dir, "annotations")
        dst_img_folder = os.path.join(parent_folder_dir, "images")
        dst_lab_folder = os.path.join(parent_folder_dir, "labels")
        for data_i in range(split_index[0], split_index[1]+1):
            current_file_name = data_names[data_i]['name']
            # Copy the annotation
            orig_path = os.path.join(orig_ann, current_file_name + ann_suffix)
            dst_path = os.path.join(dst_ann_folder, current_file_name + ann_suffix)
            copyfile(orig_path, dst_path)
            # Copy the image
            orig_path = os.path.join(orig_img, current_file_name + img_suffix)
            dst_path = os.path.join(dst_img_folder, current_file_name + img_suffix)
            copyfile(orig_path, dst_path)        
            # Copy the label
            orig_path = os.path.join(orig_lab, current_file_name + lab_suffix)
            dst_path = os.path.join(dst_lab_folder, current_file_name + lab_suffix)
            copyfile(orig_path, dst_path)

            print("Copy image {} to client {}".format(current_file_name, i))      

if __name__ == "__main__":
    orig_folders = [ORIG_ANNOT_FOLDER, ORIG_IMAGE_FOLDER, ORIG_LABEL_FOLDER]
    # Create evenly distributed sub-datasets
    evenSplit(orig_folders, OUTPUT_ROOT, data_prefix="even_subdataset")
    
    # Create non-idd sub-datasets
    noniidSplit(orig_folders, OUTPUT_ROOT, data_prefix="noniid_subdataset")

    # Just copy the validation dataset
    orig_val_folder = os.path.join(ROOT_DIR, "datasets", "VisDrone", "VisDrone2019-DET-val")
    dst_val_folder = os.path.join(ROOT_DIR, "datasets", "Validation")
    print("Copying validation dataset to {}".format(dst_val_folder))
    copytree(orig_val_folder, dst_val_folder)

    # Just copy the test dataset
    orig_test_folder = os.path.join(ROOT_DIR, "datasets", "VisDrone", "VisDrone2019-DET-test-dev")
    test_folder = os.path.join(ROOT_DIR, "datasets", "Test")
    print("Copying testing dataset to {}".format(test_folder))
    copytree(orig_test_folder, test_folder)

    print("Completed. ")