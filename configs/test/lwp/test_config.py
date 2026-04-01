import numpy as np

config = {
    "01-09_14_00_IMG_0821": {
        "mask_path": "../data/nerfosr/lwp_colmap/eval_files/eval_mask/01-09_14_00_IMG_0821.png",
        "env_map_path": "../data/nerfosr/lwp_colmap/eval_files/ENV_MAP_CC/01-09_14_00/20210901_142900.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-1/2),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 20,
        },
        "sun_angles": [-0.6 * np.pi, -0.2 * np.pi],
    },
    "24-08_11_30_IMG_9765": {
        "mask_path": "../data/nerfosr/lwp_colmap/eval_files/eval_mask/24-08_11_30_IMG_9765.png",
        "env_map_path": "../data/nerfosr/lwp_colmap/eval_files/ENV_MAP_CC/24-08_11_30/20210824_121601.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-1/2),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 20,
        },
        "sun_angles": [0.8 * np.pi, 1.2 * np.pi],
    },
    "24-08_16_30_IMG_0216": {
        "mask_path": "../data/nerfosr/lwp_colmap/eval_files/eval_mask/24-08_16_30_IMG_0216.png",
        "env_map_path": "../data/nerfosr/lwp_colmap/eval_files/ENV_MAP_CC/24-08_16_30/20210824_180014.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-1/2),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 20,
        },
        "sun_angles": [-0.3 * np.pi, 0.1 * np.pi],
    },
    "25-08_19_30_IMG_0406": {
        "mask_path": "../data/nerfosr/lwp_colmap/eval_files/eval_mask/25-08_19_30_IMG_0406.png",
        "env_map_path": "../data/nerfosr/lwp_colmap/eval_files/ENV_MAP_CC/25-08_19_30/20210825_191906.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-1/2),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 20,
        },
        "sun_angles": [-0.3 * np.pi, -0.1 * np.pi],
    },
    "31-08_07_30_IMG_0631": {
        "mask_path": "../data/nerfosr/lwp_colmap/eval_files/eval_mask/31-08_07_30_IMG_0631.png",
        "env_map_path": "../data/nerfosr/lwp_colmap/eval_files/ENV_MAP_CC/31-08_07_30/20210831_081925.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-1/2),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 20,
        },
        "sun_angles": [0.2 * np.pi, 0.7 * np.pi],
    },
}
