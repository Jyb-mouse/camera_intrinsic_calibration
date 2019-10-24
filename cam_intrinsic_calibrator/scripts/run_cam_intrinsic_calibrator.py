import os
import sys
import middleware as mw
import shutil
# set matplotlib to not use the Xwindows backend.
# matplotlib.use('Agg')

from cam_intrinsic_calibrator import CamInstrinsicCalib 

def main():
    user_config_path = os.environ[
        "TSPKG_PREFIX_PATH"] + '/share/cam_intrinsic_calibrator/configs/'
    
    rls_calibrator = CamInstrinsicCalib(user_config_path)
    rls_calibrator.img_extract()

if __name__ == "__main__":
    main()    