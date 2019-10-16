import os
import sys
import shutil
import matplotlib
# set matplotlib to not use the Xwindows backend.
matplotlib.use('Agg')

from cam_intrinsic_calibrator import RLScalibrator

def main():
    user_config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')\
    rls_calibrator = RLScalibrator(user_config_path)
    rls_calibrator.calibrate()