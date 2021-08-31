from model_features import *

def items_print(my_object, reference, output_path = None, VISU=False):
    import os
    if output_path and not os.path.isdir(output_path):
        os.makedirs(output_path)
    iterator = my_object.items()
    for k, v in iterator:
        if VISU:
            print(' ', k, ' : ', v)
        if output_path:
            with open(output_path + reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()


def construct_feature_dict()