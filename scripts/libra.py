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


        self.code_path = os.path.abspath(__file__)
        self.code_path,C = os.path.split(self.code_path)

        if not os.path.exists(self.output_path): os.makedirs(self.output_path)

        self.T_Start = time()


    ############################################################################
    ############################################################################
    def get_info_based_on_air_cnn(self):
        print("[INFO] Loading required info.")
        self.model_path = self.model_path_air
        self = get_info_from_network(self, self.model_path,
                                self.Keys_txt_file_input, self.Keys_object)


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
    def run_air_cnn(self):
        from load_models import get_network_segmentation
        from needed_functions_GPU import test_network_air

        T_Start = time()
        print("[INFO] Air segmentation using CNN is started.")
        self = get_network_segmentation(self, self.model_path,
                                        self.Keys_txt_file_input, self.Keys_object)

        self = test_network_air(self)

        T_End = time()
        print("[INFO] The total elapsed time (for all files in air CNN step): "+'\033[1m'+ \
              str(round(T_End-T_Start, 2))+'\033[0m'+" seconds")

        print("[INFO] Air segmentation using CNN is done.")


    ############################################################################
    ############################################################################
    def get_info_based_on_pec_cnn(self):
        print("[INFO] Loading required info.")
        self.model_path = self.model_path_pec
        self = get_info_from_network(self, self.model_path,
                                self.Keys_txt_file_input, self.Keys_object)


    ############################################################################
    ############################################################################
    def run_pec_preprocessing(self):
        print("[INFO] Preprocessing for breast vs pectoral segmentation.")
        T_Start = time()

        Path_segmented_air = os.path.join(self.output_path, self.saving_folder_name_net_pec_temp)

        self.Cases = sorted(glob(os.path.join(Path_segmented_air, "*"+self.image_format)))

        Image_Path = os.path.join(self.output_path, self.saving_folder_name_net_pec)
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        print("[INFO] Saving path for the summary of this step is "+Image_Path)

        for self.Case in self.Cases:
            _, File = os.path.split(self.Case)
            self.File = File[:File.find(self.air_seg_prefix)]

            call(["python3", os.path.join(self.code_path, "preprocessing_pec.py"),
                  "-i", self.Case, "-if", self.image_format, "-cn", self.File,
                  "-po", self.print_off, "-sfn", self.saving_folder_name_net_pec,
                  "-ar", str(self.A_Range), "-fis", str(self.final_image_size),
                  "-o", self.output_path])

        T_End = time()
        print("[INFO] The total elapsed time (for all files in pectroal preprocessing step): "+'\033[1m'+ \
              str(round(T_End-T_Start, 2))+'\033[0m'+" seconds")


    ############################################################################
    ############################################################################
    def run_pec_cnn(self):
        from load_models import get_network_segmentation
        from needed_functions_GPU import test_network_pec

        T_Start = time()

        print("[INFO] Pectoral segmentation using CNN is started.")
        self = get_network_segmentation(self, self.model_path,
                                        self.Keys_txt_file_input, self.Keys_object)
        self = test_network_pec(self)

        T_End = time()
        print("[INFO] The total elapsed time (for all files in pectroal CNN): "+'\033[1m'+ \
              str(round(T_End-T_Start, 2))+'\033[0m'+" seconds")

        print("[INFO] Pectoral segmentation using CNN is done.")


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
