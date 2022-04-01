# from agents.random_agent import RandomAgent
from agents.sac_agent import SACAgent


class SubmissionConfig(object):
    agent = SACAgent
    pre_eval_time = 100


class EnvConfig(object):
    multimodal = True
    eval_mode = True
    n_eval_laps = 1
    max_timesteps = 5000
    obs_delay = 0.1
    not_moving_timeout = 100
    reward_pol = "custom"
    provide_waypoints = False
    reward_kwargs = {
        "oob_penalty": 5.0,
        "min_oob_penalty": 25.0,
        "max_oob_penalty": 125.0,
    }
    controller_kwargs = {
        "sim_version": "ArrivalSim-linux-0.7.0.182276",
        "quiet": False,
        "user": "ubuntu",
        "start_container": False,
        "sim_path": "/media/antonio/Crucial 1TB/L2R_challange/ArrivalSim-linux-0.7.0.182276-l2r_chall/LinuxNoEditor",
    }
    action_if_kwargs = {
        "max_accel": 6.,
        "min_accel": -.1,
        "max_steer": 0.3,
        "min_steer": -0.3,
        "ip": "0.0.0.0",
        "port": 7077,
    }
    pose_if_kwargs = {
        "ip": "0.0.0.0",
        "port": 7078,
    }
    logger_kwargs = {
        "default": True,
    }
    cameras = {
        "CameraFrontRGB": {
            "Addr": "tcp://0.0.0.0:8008",
            "Format": "ColorBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        },
        "CameraLeftRGB": {
            "Addr": "tcp://0.0.0.0:8009",
            "Format": "ColorBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        },
        "CameraRightRGB": {
            "Addr": "tcp://0.0.0.0:8010",
            "Format": "ColorBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        },
        "CameraFrontSegm": {
            "Addr": "tcp://0.0.0.0:9008",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
        },
        "CameraLeftSegm": {
            "Addr": "tcp://0.0.0.0:9009",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
        },
        "CameraRightSegm": {
            "Addr": "tcp://0.0.0.0:9010",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
        },
        "CameraBirdsEye": {
            "Addr": "tcp://0.0.0.0:10008",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
        },
        "CameraBirdsEyeSegm": {
            "Addr": "tcp://0.0.0.0:10009",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
        },
    }


class SimulatorConfig(object):
    racetrack = "Thruxton"
    active_sensors = [
        "CameraFrontRGB",
    ]
    driver_params = {
        "DriverAPIClass": "VApiUdp",
        "DriverAPI_UDP_SendAddress": "0.0.0.0",
    }
    camera_params = {
        "Format": "ColorBGR8",
        "FOVAngle": 90,
        "Width": 512,
        "Height": 384,
        "bAutoAdvertise": True,
    }
    vehicle_params = False
