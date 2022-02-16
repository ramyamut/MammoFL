#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore")

from time import time
from glob import glob
from copy import deepcopy
from subprocess import call
import os, pdb, multiprocessing

from get_info import get_info_from_network
from initialize_variables import set_argparse, get_variables


################################################################################
################################################################################
class LIBRA(object): # The main class
    def __init__(self):
        self.version = "version-1.0"


    ############################################################################
    ############################################################################
    def parse_args(self, argv=None):
        args = set_argparse(argv)
        self = get_variables(self, args)

        self.number_cpu_cores = 1
        self.core_multiplier = 1
        self.number_of_threads = 1

        self.number_cpu_cores = self.number_cpu_cores*self.number_of_threads
        self.max_number_of_process = int(self.core_multiplier*self.number_cpu_cores)


        self.Keys_txt_file_input = ['image_format', 'num_class', 'save_period',
                                    'model', 'backbone', 'training_mode',
                                    'flag_multi_class', 'A_Range', 'image_final_size']
        self.Keys_object = self.Keys_txt_file_input #### this is to name the output keys/ keep it the same


        self.saving_folder_name_net_pec_temp, folder_name = os.path.split(self.saving_folder_name_net_pec)
        self.saving_folder_name_net_pec_temp = self.saving_folder_name_net_pec_temp+"_temp"
        self.saving_folder_name_net_pec_temp = os.path.join(self.saving_folder_name_net_pec_temp, folder_name)


        if self.num_gpu == 0:
            self.test_batch_size = 1
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""


        self.batch_size = self.test_batch_size
        self.image_final_size = self.final_image_size
        self.image_format = '.png'


        self.code_path = os.path.abspath(__file__)
        self.code_path,C = os.path.split(self.code_path)

        if not os.path.exists(self.output_path): os.makedirs(self.output_path)

        self.T_Start = time()


    ############################################################################
    ############################################################################
    def run_air_preprocessing(self):
        print("[INFO] Air segmentation preprocessing.")

        T_Start = time()

        self.Cases = sorted(glob(os.path.join(self.input_data, "*dcm")))

        Image_Path = os.path.join(self.output_path, self.saving_folder_name_net_air)
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        print("[INFO] Saving path for the summary of this step is "+Image_Path)

        for self.Case in self.Cases:
            Path, File = os.path.split(self.Case)
            File = File[:-4]

            call(["python3", os.path.join(self.code_path, "preprocessing.py"), "-i",
                  self.Case, "-o", self.output_path, "-if", self.image_format,
                  "-po", self.print_off, "-sfn", self.saving_folder_name_net_air,
                  "-ar", str(self.A_Range), "-fis", str(self.final_image_size),
                  "-lsm", self.libra_segmentation_method, "-fpm", self.find_pacemaker])

        T_End = time()
        print("[INFO] The total elapsed time (for all files in air preprocessing step): "+'\033[1m'+ \
              str(round(T_End-T_Start, 2))+'\033[0m'+" seconds")


    ############################################################################
    ############################################################################
    def run_breast_postprocessing(self):
        print("[INFO] Postprocessing for breast vs pectroal segmentation.")
        T_Start = time()

        Path_segmented_pectoral = os.path.join(self.output_path, self.saving_folder_name_temp_breast_masks)

        self.Cases = sorted(glob(os.path.join(Path_segmented_pectoral, "*"+self.image_format)))

        Image_Path = os.path.join(self.output_path, self.saving_folder_name_final_masked_normalized_images)
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        print("[INFO] Saving path for the summary of this step is "+Image_Path)

        for self.Case in self.Cases:
            _, File = os.path.split(self.Case)
            self.File = File[:File.find(self.pec_seg_prefix)]

            call(["python3", os.path.join(self.code_path, "postprocessing.py"),
                  "-i", self.Case, "-if", self.image_format, "-cn", self.File,
                  "-po", self.print_off, "-sfn", self.saving_folder_name_final_masked_normalized_images,
                  "-ar", str(self.A_Range), "-fis", str(self.final_image_size),
                  "-o", self.output_path, "-fb", self.find_bottom])

        T_End = time()
        print("[INFO] The total elapsed time (for all files in breast postprocessing step): "+'\033[1m'+ \
              str(round(T_End-T_Start, 2))+'\033[0m'+" seconds")
