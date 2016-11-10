import argparse
from datetime import datetime
from glob import glob
import os
import sys
import yaml
import numpy
import tensorflow as tf

from chen_huang.datasets.data_loaders import LandmarkDataLoader
from chen_huang.data_batchers.video_seq_generator import FeatureSequenceGenerator
from chen_huang.models.recurrent_model import RecurrentModel
from chen_huang.util.normalizers import LandmarkFeatureNormalizer


def make_mask(seq_lengths, batch_size, max_sequence_length):
    mask = numpy.zeros((batch_size, max_sequence_length), numpy.bool)
    
    for i in range(batch_size):
        mask[i, 0:seq_lengths[i]] = 1
    
    return mask


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_dict = yaml.load(f)
        
    return yaml_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Script to pick best performing model on val set.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                         help='Directory containing checkpoint (.ckpt) files.')
    parser.add_argument('--yaml_file', dest='yaml_file',
                        help='yaml file containing hyperparameters.',
                        default='./params.yaml')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Load data, model, and training parameters
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir 
    yaml_file = args.yaml_file #sys.argv[1] #'./params.yaml'
    param_dict = load_yaml(yaml_file)
    data_dict = param_dict['data']
    model_dict = param_dict['model']
    train_dict = param_dict['train']

    dataset_path = data_dict['dataset_path']
    feat_type = data_dict['feat_type']
    fold_type = data_dict['fold_type']
    train_folds = data_dict['train_folds']
    val_folds = data_dict['val_folds']
    test_folds = data_dict['test_folds']
    remove_labels = data_dict['remove_labels']
    if 'remove_easy' in data_dict.keys():
        remove_easy = data_dict['remove_easy']
    else:
        remove_easy = False

    batch_size = model_dict['batch_size']
    max_sequence_length = model_dict['max_sequence_length']

    # Load datasets
    data_loader_train = LandmarkDataLoader(dataset_path,
                                           feat_type=feat_type, 
                                           fold_type=fold_type,
                                           which_fold=train_folds,
                                           remove_easy=remove_easy)
    data_dict_train = data_loader_train.load()


    data_loader_val = LandmarkDataLoader(dataset_path,
                                         feat_type=feat_type, 
                                         fold_type=fold_type,
                                         which_fold=val_folds,
                                         remove_easy=remove_easy)
    data_dict_val = data_loader_val.load()


    data_loader_test = LandmarkDataLoader(dataset_path,
                                          feat_type=feat_type, 
                                          fold_type=fold_type,
                                          which_fold=test_folds,
                                          remove_easy=remove_easy)
    data_dict_test = data_loader_test.load()


    X_train = data_dict_train['X']
    y_train = data_dict_train['y']

    X_val = data_dict_val['X']
    y_val = data_dict_val['y']

    X_test = data_dict_test['X']
    y_test = data_dict_test['y']

    # Load data batchers
    batcher_train = FeatureSequenceGenerator(X_train, y_train, batch_size=batch_size,
                                             max_seq_length=max_sequence_length,
                                             verbose=False)

    batcher_val = FeatureSequenceGenerator(X_val, y_val, batch_size=batch_size,
                                           max_seq_length=max_sequence_length,
                                           verbose=False)

    batcher_test = FeatureSequenceGenerator(X_test, y_test, batch_size=batch_size,
                                           max_seq_length=max_sequence_length,
                                           verbose=False)

    # Load preprocessors
    normalizer = LandmarkFeatureNormalizer(X_train)

    # Load model
    model = RecurrentModel(model_dict, verbose=True)

    # Evaluate on training set
    #x_train_all, y_train_all, seq_lengths_train_all = batcher_train.fetch_all_samples()
    #print x_train_all.shape, y_train_all.shape, seq_lengths_train_all.shape

    #y_batch = numpy.tile(y_train_all[:, None], (1, max_sequence_length))
    #mask = make_mask(seq_lengths_train_all, 220, max_sequence_length)
    #x_batch_norm = normalizer.run(x_train_all)
        
    #accuracy_train_clip = model.get_accuracy_clip(x_batch_norm, y_batch, seq_lengths_train_all, mask)
    #print accuracy_train_clip

    # Evaluate on validation set
    x_val_all, y_val_all, seq_lengths_val_all = batcher_val.fetch_all_samples()
    print x_val_all.shape, y_val_all.shape, seq_lengths_val_all.shape

    y_batch = numpy.tile(y_val_all[:, None], (1, max_sequence_length))
    mask_val = make_mask(seq_lengths_val_all.astype('int'), 220, max_sequence_length)
    x_batch_norm = normalizer.run(x_val_all)

    checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, '*.ckpt')))
    print checkpoint_files
    
    accuracies_val = []
    for i, checkpoint_file in enumerate(checkpoint_files):
        print 'Loading checkpoint: %s' % checkpoint_file
        model.load(checkpoint_file)
        val_accuracy = model.get_accuracy_clip(x_batch_norm, y_batch, seq_lengths_val_all, mask_val)
        print 'Val Accuracy: %f' % val_accuracy
        accuracies_val.append(val_accuracy)

    accuracies_val = numpy.array(accuracies_val)
    accuracies_val = numpy.around(accuracies_val, 5)
    max_val_accuracy = numpy.max(accuracies_val)
    max_checkpoint_ind = numpy.argmax(accuracies_val)
    max_checkpoint = checkpoint_files[max_checkpoint_ind]
    # print 'Accuracies_val: %s' % accuracies_val
    print 'Checkpoint_ind: %d' % max_checkpoint_ind
    print 'Max Checkpoint: %s' % max_checkpoint
    print 'Max Val Accuracy: %f' % max_val_accuracy

    # Evaluate on test set
    x_test_all, y_test_all, seq_lengths_test_all = batcher_test.fetch_all_samples()
    print x_test_all.shape, y_test_all.shape, seq_lengths_test_all.shape

    y_batch = numpy.tile(y_test_all[:, None], (1, max_sequence_length))
    mask_test = make_mask(seq_lengths_test_all.astype('int'), 220, max_sequence_length)
    x_batch_norm = normalizer.run(x_test_all)

    #inds = range(0, 220)
    #accuracy_test_clip = model.get_accuracy_clip(x_batch_norm[inds, :, :], y_batch[inds], seq_lengths_test_all[inds], mask_test[inds, :])
    #print accuracy_test_clip

    model.load(max_checkpoint)
    test_accuracy = model.get_accuracy_clip(x_batch_norm, y_batch, seq_lengths_test_all, mask_test)
    print 'Test Accuracy: %f' % test_accuracy
 
