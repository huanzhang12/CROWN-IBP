## utils.py
## 
## Several utility functions
##
## Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
## See CREDITS for a list of contributors.
##

import numpy as np
import random
import os
from PIL import Image


def linf_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=np.inf)

def l2_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=2)

def l1_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=1)

def l0_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=0)

def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    return
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def binary_search(cond, current, upper = np.inf, lower = 0.0, tol = 0.00001, max_steps = 100, upscale_mult = 10.0, downscale_mult = 10.0, upper_limit = 100000.0):
    upper_not_found = True
    lower_not_found = True
    step = 0
    while upper-lower > tol:
        success, val = cond(current)
        if val is not None:
            print("[L2][binary search] step = {}, current = {:.6f}, success = {}, val = {:.2f}".format(step,current,success,val))
        else:
            print("[L2][binary search] step = {}, current = {:.6f}, success = {}".format(step,current,success))
        if success: # success at current value
            if upper_not_found: # always success, we have not found the true upper bound
                # but this is a valid lower bound
                lower = current
                # increase search range, try to find an upper bound
                current *= upscale_mult
            else:
                lower = current
                current = 0.5 * (lower + upper)
            # when initial is a success we always have a valid lower bound
            lower_not_found = False
        else:
            if lower_not_found: # so far always < 0, haven't found eps_LB
                upper = current
                current /= downscale_mult
            else:
                upper = current
                current = 0.5 * (lower + upper)
            # when initial is a failure we always have a valid upper bound
            upper_not_found = False
        step += 1
        if step >= max_steps:
            break
        if current > upper_limit:
            break
    return lower

def generate_data(data, samples, targeted=True, random_and_least_likely = False, skip_wrong_label = True, 
                  force_label = None, start=0, ids = None, target_classes = None, target_type = 0b1111, 
                  predictor = None, total_images = -1, imagenet=False, remove_background_class=False, 
                  save_inputs=False, model_name=None, save_inputs_dir=None):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    ids: true IDs of images in the dataset, if given, will use these images
    target_classes: a list of list of labels for each ids
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    true_labels = []
    true_ids = []
    information = []
    target_candidate_pool = np.eye(data.test_labels.shape[1])
    target_candidate_pool_remove_background_class = np.eye(data.test_labels.shape[1] - 1)
    print('generating labels...')
    print('selecting {} images from {} candidates'.format(total_images if total_images else samples, samples))
    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        if target_classes:
            if type(target_classes) is list:
                target_classes = target_classes[start:start+samples]
        start = 0
    if total_images == -1:
        total_images = samples
    total = 0
    n_correct = 0
    n_image_added = 0
    for i in ids:
        total += 1
        image_added = False
        if targeted:
            predicted_label = -1 # unknown
            if random_and_least_likely:
                original_predict = np.squeeze(predictor(np.array([data.test_data[start+i]])))
                num_classes = len(original_predict)
                predicted_label = np.argmax(original_predict)
                true_label = np.argmax(data.test_labels[start+i])
                # if there is no user specified target classes
                if target_classes is None:
                    least_likely_label = np.argmin(original_predict)
                    runnerup_label = np.argsort(original_predict)[-2]
                    start_class = 1 if (imagenet and not remove_background_class) else 0
                    random_class = predicted_label
                    new_seq = [least_likely_label, runnerup_label, predicted_label]
                    while random_class in new_seq:
                        random_class = random.randint(start_class, start_class + num_classes - 1)
                    new_seq[2] = random_class
                    seq = []
                    if true_label != predicted_label and skip_wrong_label:
                        seq = []
                    elif force_label is not None:
                        # true_label == predicted_label
                        if true_label != force_label:
                            seq = []
                        else:
                            seq.append(true_label)
                            information.append(str(force_label))
                            image_added = True
                    else:
                        image_added = True
                        if target_type & 0b10000:
                            for c in range(num_classes):
                                if c != predicted_label:
                                    seq.append(c)
                                    information.append('class'+str(c))
                        else:
                            if target_type & 0b0100:
                                # least
                                seq.append(new_seq[0])
                                information.append('least')
                            if target_type & 0b0001:
                                # top-2
                                seq.append(new_seq[1])
                                information.append('runnerup')
                            if target_type & 0b0010:
                                # random
                                seq.append(new_seq[2])
                                information.append('random')
                else:
                    if true_label != predicted_label and skip_wrong_label:
                        seq = []
                    else:
                        if (force_label is not None) and (true_label == force_label):
                            image_added = True
                            # use user specified target classes
                            if type(target_classes) is list:
                                seq = target_classes[total - 1]
                            else:
                                seq = [target_classes]
                            information.extend(len(seq) * ['user'])
                        else:
                            seq = []
            else:
                image_added = True
                if imagenet:
                    if remove_background_class:
                        seq = random.sample(range(0,1000), 10)
                    else:
                        seq = random.sample(range(1,1001), 10)
                    information.extend(data.test_labels.shape[1] * ['random'])
                else:
                    seq = range(data.test_labels.shape[1])
                    information.extend(data.test_labels.shape[1] * ['seq'])
            is_correct = np.argmax(data.test_labels[start+i]) == predicted_label
            if is_correct:
                n_correct += 1
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, force_label = {}, correct = {}, seq = {}, info = {}".format(total, start + i, 
                  np.argmax(data.test_labels[start+i]), predicted_label, force_label, is_correct, seq, [] if len(seq) == 0 else information[-len(seq):]))
            for j in seq:
                # skip the original image label
                if not force_label and (j == np.argmax(data.test_labels[start+i])):
                    continue
                inputs.append(data.test_data[start+i])
                if remove_background_class:
                    targets.append(target_candidate_pool_remove_background_class[j])
                else:
                    targets.append(target_candidate_pool[j])
                true_labels.append(data.test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
        else:
            true_label = np.argmax(data.test_labels[start+i])
            original_predict = np.squeeze(predictor(np.array([data.test_data[start+i]])))
            num_classes = len(original_predict)
            predicted_label = np.argmax(original_predict) 
            is_correct = np.argmax(data.test_labels[start+i]) == predicted_label
            if is_correct:
                n_correct += 1
            if true_label != predicted_label and skip_wrong_label:
                pass
            else:
                image_added = True
                inputs.append(data.test_data[start+i])
                if remove_background_class:
                    # shift target class by 1
                    print(np.argmax(data.test_labels[start+i]))
                    print(np.argmax(data.test_labels[start+i][1:1001]))
                    # targets.append(data.test_labels[start+i][1:1001])
                    targets.append(target_candidate_pool[predicted_label])
                else:
                    # targets.append(data.test_labels[start+i])
                    targets.append(target_candidate_pool[predicted_label])
                true_labels.append(data.test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
                information.extend(['original'])
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, added = {}".format(total, start + i, true_label, predicted_label, is_correct, image_added))
        if image_added:
            n_image_added += 1
            if n_image_added >= total_images:
                break

    inputs = np.array(inputs)
    targets = np.array(targets)
    true_labels = np.array(true_labels)
    true_ids = np.array(true_ids)
    print('labels generated')
    print('{} images generated in total.'.format(len(inputs)))
    print('top-1 accuracy:', n_correct / float(total))
    if save_inputs:
        if not os.path.exists(save_inputs_dir):
            os.makedirs(save_inputs_dir)
        save_model_dir = os.path.join(save_inputs_dir,model_name)
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        info_set = list(set(information))
        for info_type in info_set:
            save_type_dir = os.path.join(save_model_dir,info_type)
            if not os.path.exists(save_type_dir):
                os.makedirs(save_type_dir)
            counter = 0
            for i in range(len(information)):
                if information[i] == info_type:
                    df = inputs[i,:,:,0]
                    df = df.flatten()
                    np.savetxt(os.path.join(save_type_dir,'point{}.txt'.format(counter)),df,newline='\t')
                    counter += 1
            target_labels = np.array([np.argmax(targets[i]) for i in range(len(information)) if information[i]==info_type])
            np.savetxt(os.path.join(save_model_dir,model_name+'_target_'+info_type+'.txt'),target_labels,fmt='%d',delimiter='\n') 
    return inputs, targets, true_labels, true_ids, information

