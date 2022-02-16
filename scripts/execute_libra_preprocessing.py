import warnings
warnings.filterwarnings("ignore")

#!/usr/bin/python3
from time import time
from libra import *
from initialize_variables import set_argparse, get_variables


# python3 ~/github/LIBRA/final/execute_libra_preprocessing.py -i ~/comp_space/dataset/ -o ~/comp_space/dataset/libra_new2


class run_libra(object):
    def __init__(self):
        args = set_argparse(argv=None)
        self = get_variables(self, args)


    def main_function(self):
        Info = LIBRA()
        print("[INFO] Starting LIBRA "+Info.version)


        Info.parse_args(["-i", self.input_data,
                        "-po", self.print_off,
                         "-o", self.output_path,
                         "-ng", str(self.num_gpu),
                         "-mc", str(self.multi_cpu),
                         "-fb", str(self.find_bottom),
                         "-m", self.general_model_path,
                         "-lt", str(self.libra_training),
                         "-cm", str(self.core_multiplier),
                         "-fpm", str(self.find_pacemaker),
                         "-tow",str(self.timeout_waiting),
                         "-tbs", str(self.test_batch_size),
                         "-fis", str(self.final_image_size),
                         "-not", str(self.number_of_threads),
                         "-wsm", self.weight_selection_method,
                         "-wttbd", self.which_task_to_be_done,
                         "-rii", self.remove_intermediate_images,
                         "-lsm", str(self.libra_segmentation_method)])


        Info.run_breast_preprocessing()

        T_End = time()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    RUN = run_libra()
    RUN.main_function()
