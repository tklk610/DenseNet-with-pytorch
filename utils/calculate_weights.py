import os
from tqdm import tqdm
import numpy as np
from dataloaders import cfg

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for i, (image, label) in tqdm_batch:
        y = sample['label']
        label = label.detach().cpu().numpy()
        mask    = (label >= 0) & (label < num_classes)
        labels  = label[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z      += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights   = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(cfg.DATASET_DIR.db_root_dir(dataset), dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret