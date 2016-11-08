import argparse
from datetime import datetime
import os
import sys
import yaml
import numpy
import tensorflow as tf
# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

from chen_huang.datasets.data_loaders import LandmarkDataLoader
from chen_huang.data_batchers.video_seq_generator import FeatureSequenceGenerator
from chen_huang.models.recurrent_model import RecurrentModel
from chen_huang.util.normalizers import LandmarkFeatureNormalizer
# from chen_huang.util.data_augmenter import DataAugmenter


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
    parser = argparse.ArgumentParser(description='Script to train RNN/LSTM models. ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

    # Train Loop
    total_num_steps = train_dict['total_num_steps']
    disp_step = train_dict['disp_step']
    val_step = train_dict['val_step']
    save_step = train_dict['save_step']

    for step in range(total_num_steps):
        # Load batch from training set
        x_batch, y_batch_temp, seq_length_batch = batcher_train.get_batch()
        y_batch = numpy.tile(y_batch_temp[:, None], (1, max_sequence_length))
        mask = make_mask(seq_length_batch, batch_size, max_sequence_length)
        x_batch_norm = normalizer.run(x_batch)
        
        cost_train, accuracy_train, accuracy_train_clip, summary_train = model.train(x_batch_norm,
                                                                                     y_batch,
                                                                                     seq_length_batch,
                                                                                     mask)
        #accuracy_train_clip = model.get_accuracy_clip(x_batch_norm, y_batch, seq_length_batch, mask)
        
        if step % disp_step == 0:
            print 'Iter %d --- batch_cost_train: %.4f --- batch_accuracy_train: %.4f --- clip_accuracy_train: %.4f' % (step, cost_train, accuracy_train, accuracy_train_clip)

        if step % 10 == 0:
            #print 'Added to train summaries'    
            model.summary_writer_train.add_summary(summary_train, step)

        if step % val_step == 0:
            x_batch_val, y_batch_temp_val, seq_length_batch_val = batcher_val.get_batch()
            y_batch_val = numpy.tile(y_batch_temp_val[:, None], (1, max_sequence_length))
            mask_val = make_mask(seq_length_batch_val, batch_size, max_sequence_length)
            x_batch_norm_val = normalizer.run(x_batch_val)
                    
            #cost_val = model.cost(x_batch_norm_val, y_batch_val, seq_length_batch_val, mask_val)
            #accuracy_val = model.get_accuracy(x_batch_norm_val, y_batch_val, seq_length_batch_val, mask_val)
            #accuracy_val_clip = model.get_accuracy_clip(x_batch_norm_val, y_batch_val, seq_length_batch_val, mask_val)
            cost_val, accuracy_val, accuracy_val_clip, summary_val = model.val_batch(x_batch_norm_val,
                                                                                     y_batch_val,
                                                                                     seq_length_batch_val,
                                                                                     mask_val)
 
            print '*Iter %d --- batch_cost_val: %.4f --- batch_accuracy_val: %.4f --- clip_accuracy_val: %.4f' % (step, cost_val, accuracy_val, accuracy_val_clip)

        if step % val_step == 0:
            #print 'Added to val summaries'
            model.summary_writer_val.add_summary(summary_val, step)

        if step % save_step == 0:
            dt = datetime.now()
            time_str = dt.strftime("%mm-%dd-%yy-%Hh-%Mm-%Ss")
            model.save('ckpt_'+time_str)


