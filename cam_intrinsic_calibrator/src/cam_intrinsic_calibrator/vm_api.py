import os
import sys
import yaml
import tarfile
import base64
import re
import middleware as mw

from subprocess import check_output, CalledProcessError
from ts_rpc.client import RemoteObject

TSPKGNAME = 'vehicle_config_{}'
COMMIT_MSGS_FORMAT = "Update {} {} calibration"


class VM(object):

    def __init__(self, api_key, vehicle, reviewer, url, results_file_path, intrinsic_cam_id):

        self._api_key = api_key
        self._vehicle = vehicle
        self._reviewer = reviewer
        self._vm = RemoteObject(str(url))
        self._tspkg_name = TSPKGNAME.format(self._vehicle.lower())
        self._results_file_path = results_file_path
        self._intrinsic_cam_id = intrinsic_cam_id
        self._is_vm_has_vehicle = False
        self._is_local_has_vehicle = False

        self._init_vm()

    # update vehicle config
    def _init_vm(self):
        # install vehicle config by vehicle name
        if self._vehicle.startswith('Octopus-'):
            self._vehicle = self._vehicle[len('Octopus-'):]
        
        vm_pkg_list_str = str(check_output('tsp list -a --yaml 2> /dev/null', shell=True).decode('utf-8')).strip()
        if "vehicle_config_{}".format(self._vehicle.lower()) in vm_pkg_list_str:
            self._is_vm_has_vehicle = True
        local_pkg_list_str = str(check_output('tsp list --yaml 2> /dev/null', shell=True).decode('utf-8')).strip()
        if "vehicle_config_{}".format(self._vehicle.lower()) in local_pkg_list_str:
            self._is_local_has_vehicle = True
        
        # upgrade vehicle config
        if self._is_vm_has_vehicle == True and self._is_local_has_vehicle == True:
            os.system("tsp upgrade vehicle_config_{} -y".format(
                self._vehicle.lower()))
        elif self._is_vm_has_vehicle == True and self._is_local_has_vehicle == False:
            os.system("tsp install vehicle_config_{} -y".format(
                self._vehicle.lower()))
        else:
            #raise Exception('\n---------This is a new vehicle and the VM cannot be updated!!! ---------\n')
            print('\n---------This is a new vehicle and the VM cannot be updated!!! ---------\n')

    # update extrinsic by config.yaml
    def update_extrinsic(self):
        extrinsic_lst = self._get_extrinsic_pairs()
        extrinsic_path = self._get_calib_path() + '/extrinsic/'

        for f_name in os.listdir(extrinsic_path):
            if f_name in extrinsic_lst:
                extri_data = open(extrinsic_path+f_name, 'r').read()
                extri_dest = 'calibration/extrinsic/'+f_name
                result = self._vm.api.updateFile(
                    self._api_key, TSPKGNAME.format(self._vehicle.lower()), extri_data, extri_dest)

        # msg = COMMIT_MSGS_FORMAT.format(self._vehicle, 'extrinsic')
        # commit_ret = self.commit(msg)
        # return commit_ret

    # update intrinsic by config.yaml
    def update_intrinsic(self):
        intrinsic_lst = self._get_intrinsic_sensors()
        intrinsic_path = self._results_file_path + '/{}/cam_{}/'.format(self._vehicle, self._intrinsic_cam_id)
        for f_name in os.listdir(intrinsic_path):
            if f_name in intrinsic_lst:
                intri_data = open(intrinsic_path+f_name, 'r').read()
                intri_dest = 'calibration/intrinsic/'+f_name
                result = self._vm.api.updateFile(
                    self._api_key, TSPKGNAME.format(self._vehicle.lower()), intri_data, intri_dest)
        # msg = COMMIT_MSGS_FORMAT.format(self._vehicle, 'intrinsic')
        # commit_ret = self.commit(msg)
        # return commit_ret

    # update config.yaml
    def update_config_yaml(self):
        self._cfg_file = self._get_calib_path()+'/config.yaml'
        cfg_data = open(self._cfg_file, 'r').read()
        cfg_dest = 'calibration/config.yaml'
        result = self._vm.api.updateFile(
            self._api_key, self._vehicle.lower(), cfg_data, cfg_dest)
        return [True, result['result']] if result['result'] == 'success' else [False, result['result']]

    # commit change and add reviewer
    def commit(self, msg):
        result = self._vm.api.submitChange(self._api_key, TSPKGNAME.format(self._vehicle.lower()),
                                           msg, self._reviewer)
        return result

    # get cam_type
    def get_cam_type(self):
        cam_type = dict()
        vehicle_cfg = self._get_vehicle_cfg()
        for sensor, param in vehicle_cfg.get('components').items():
            if 'camera' in sensor:
                cam_type.update(
                    {int(re.findall(r'\d+', sensor)[0]): param['model']})
        return cam_type

    # get calibration config.yaml
    def _get_calib_config(self):
        self._cfg_file = self._get_calib_path()+'/config.yaml'
        with open(self._cfg_file, "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
        return cfg['configs']

    def _get_calib_path(self):
        # get calibration dir
        return self._get_vm_path()+'/calibration'

    def _get_vehicle_cfg(self):
        # get calibration dir
        with open(os.path.join(self._get_vm_path() + '/vehicle.yaml')) as vf:
            try:
                return yaml.safe_load(vf)
            except yaml.YAMLError as e:
                print("Cannot get vehicle.yaml from {} of VM".format(self._vehicle))
                raise(e)

    # get VM path
    def _get_vm_path(self):
        try:
            #return str(self._results_file_path + '/vehicle_config_{}/res'.format(self._vehicle.lower()))
            return str(check_output('tsp show --path vehicle_config_{} 2> /dev/null'.format(self._vehicle.lower()), shell=True).decode('utf-8')).strip() + '/res'
        except CalledProcessError as e:
            print(e)

    # get intrinsic sensors from config.yaml
    def _get_intrinsic_sensors(self):
        cfg = self._get_calib_config()
        intrinsic_lst = []

        for key, value in cfg.items():
            for dest_sensor, path in value.iteritems():
                if dest_sensor == 'intrinsic':
                    intrinsic_lst.append(path)
        return intrinsic_lst

    # get intrinsic sensors from config.yaml
    def _get_extrinsic_pairs(self):
        cfg = self._get_calib_config()
        extrinsic_lst = []
        for key, value in cfg.items():
            for dest_sensor, path in value.items():
                if dest_sensor == 'intrinsic':
                    continue
                else:
                    extrinsic_lst.append(path[0]+'_to_'+path[1]+'.yaml')
        return extrinsic_lst

# for test and example 
# def main():
#     # api_key = "bin.chao:c028560da5f37b71bd1f23fd55c524c1c03a84c502e3c7d8839e3cfbbcf22263"
#     api_key = 'bin.chao:c69a0874434141f7e37cd824b02482a82c62224da2fc0d460b3ed984b159bb60'
#     vehicle = 'shaanqi-2003'
#     reviewer = "bin.chao"
#     url = "http://vm2.bj.tusimple.ai:5024"
#     file_path = '/root/.tspkg/share/cam_intrinsic_calibrator/results'

#     vm = VM(api_key, vehicle, reviewer, url, file_path, 1)

#     # print vm._get_calib_path()
#     # print vm._get_calib_cfg_path()
#     # print vm._get_intrinsic_sensors()
#     # print vm._get_extrinsic_pairs()

#     print(vm.update_config_yaml())
#     # vm._get_extrinsic_pairs()
#     print(vm.update_extrinsic())
#     vm.update_intrinsic()
#     # vm.commit()
#     # vm.get_workspace()
#     # # print(vm._get_vehicle_cfg())
#     # print(vm.get_sensor_layout())

# if __name__ == "__main__":
#     main()