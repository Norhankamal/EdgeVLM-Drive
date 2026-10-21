# EdgeVLM-Drive

Driver monitoring on a **Jetson Nano 4GB** with a small vision-language model (SmolVLM-256M or SmolVLM-500M, fine-tuned with LoRA and merged). The system looks at the driver and answers with one label: `NORMAL`, `DROWSY`, `LOOKING_AWAY` or `PHONE_USAGE`. When it sees `DROWSY`, it raises an alert.

It is built as a ROS 2 (Foxy) workspace. A face mesh (MediaPipe) can be switched on to skip frames that are clearly normal.

```
camera_node ──/dms/camera/raw──► preprocess_node ──/dms/frame_pair──► inference_node ──/dms/state──► alert_node ──/dms/alert──► alarm on laptop
 (webcam)                        enhance image                        SmolVLM                        DROWSY? alert
                                 mesh gate (optional)                    │
                                 full crop + face crop                   └─/benchmark/raw_metrics──► post_benchmark_node ──► dms_benchmark.csv
```

## Repository layout

```
ros2_dms_code/                      <- the ROS 2 workspace (clone the repo into this folder name)
└── src/
    ├── dms_interfaces/             message definition (FramePair.msg)
    └── dms_ros/
        ├── config/dms_params.yaml  all settings of the nodes
        ├── config/prompts.yaml     one training prompt per model (do not change the text)
        ├── launch/dms_launch.py    starts the 5 nodes
        ├── dms_ros/                the ROS nodes (installed Python package)
        ├── validate.py             test of the 1200 images, model only (no ROS)
        ├── mesh_gate_eval.py       test of the 1200 images, with the mesh gate (no ROS)
        ├── dms_preprocess.py       image preprocessing and mesh gate rules (used by the two scripts)
        ├── mesh_landmark_extractor.py
        └── validation_results/     results of the 1200-image tests
```

`dms_preprocess.py` and `mesh_landmark_extractor.py` exist **twice**: once next to `validate.py` (for the scripts) and once inside `dms_ros/dms_ros/` (for the installed nodes). If you edit one, copy it to the other.

## Paths used in this guide

| What | Path |
| --- | --- |
| Workspace | `~/ros2_dms_code` |
| Models | `/home/jetson/models/SmolVLM-256M-Finetuned` and `/home/jetson/models/SmolVLM-500M-Finetuned` |
| Test images (1200) | `/home/jetson/test_dataset/<CLASS>/` |
| Mesh model file | `~/.cache/mediapipe/face_landmarker.task` |
| Benchmark file of a live run | `/home/jetson/dms_benchmark.csv` |

The user name on the Jetson is `jetson`.

---

## 1. Install Ubuntu 20.04 from Q-engineering

You need a Jetson Nano 4GB developer kit and an SD card of **64 GB or more** (32 GB is the minimum, but the installed software uses more than 21 GB). For MAXN power mode (used for the live runs below), use a 5 V 4 A barrel power supply.

1. Open the image page: <https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image>
2. Download `JetsonNanoUb20_3b.img.xz` (about 9 GB). Check the MD5 value on that page after the download.
3. Flash the `.xz` file **directly** (do not unpack it) to the SD card with balenaEtcher or Raspberry Pi Imager.
4. Put the card in the Jetson, connect a monitor, keyboard and network, and boot. The password on the image is `jetson`. Use `jetson` as the user name.
5. When Ubuntu offers it, let the Software Updater finish.
6. Use the whole card: install GParted (`sudo apt install gparted`) and enlarge the main partition.
7. Check:
   ```
   python3 --version
   python3 -c "import torch, cv2; print(torch.__version__, cv2.__version__, torch.cuda.is_available())"
   ```
   You should see torch 1.13.0, OpenCV 4.8.0 and `True`.

The image already contains Ubuntu 20.04, OpenCV 4.8.0, PyTorch 1.13.0, TorchVision 0.14.0, TensorRT 8.0.1.6, TensorFlow 2.4.1 and jtop 4.2.1.

**Config in `dms_params.yaml`:** none yet.

## 2. Extra dependencies and versions

### ROS 2 Foxy

ROS 2 Foxy is not on the image. It is an old release, but this project was built and tested on it. Follow the official Foxy install page on docs.ros.org (deb packages). In short:

```
sudo apt update && sudo apt install -y software-properties-common curl git
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-foxy-ros-base python3-colcon-common-extensions
```

Check: `source /opt/ros/foxy/setup.bash && printenv ROS_DISTRO` prints `foxy`.

### Python packages

| Package | Version | Used for |
| --- | --- | --- |
| Python | 3.8.10 | (Ubuntu 20.04) |
| CUDA | 10.2 (V10.2.300) | on the image |
| TensorRT | 8.0.1.6 | on the image |
| GLIBC | 2.31 | Ubuntu 20.04 |
| Kernel | 4.9.253-tegra | |
| psutil | 7.2.2 | RAM monitor |
| torch | 1.13.0a0+git7c98e70 | on the image |
| torchvision | 0.14.0a0+5ce4506 | on the image |
| OpenCV (`cv2`) | 4.8.0 (image). `pip` shows `opencv-python-headless` 5.0.0.93 | preprocessing and face detector |
| transformers | 4.46.0 | SmolVLM |
| accelerate | 1.0.1 | needed by `low_cpu_mem_usage=True` when loading the model |
| tokenizers | 0.20.3 | transformers |
| safetensors | 0.5.3 | transformers |
| huggingface-hub | 0.23.2 | transformers |
| numpy | 1.18.5 | |
| Pillow | 9.5.0 | images for the processor |
| jetson-stats (jtop) | 4.2.1 (on the image) | optional, RAM numbers in the benchmark node; without it the node reads `/proc/meminfo` |
| Jetson.GPIO | optional | only if you connect a buzzer (`buzzer_pin`) |

Install what is missing (use the versions in the table):

```
pip3 install transformers==4.46.0 accelerate==1.0.1 tokenizers==0.20.3 safetensors==0.5.3 huggingface-hub==0.23.2 pillow==9.5.0 psutil pandas scikit-learn tqdm pyyaml
```

**MediaPipe on the Jetson.** The normal `mediapipe` wheel from PyPI may not work on the Jetson (aarch64). Install a build that works on your board (for example from the mediapipe-bin project, see the note at the top of `mesh_landmark_extractor.py`), and check:

```
python3 -c "import mediapipe; print(mediapipe.__version__)"
```

**Mesh model file** (once per Jetson):

```
mkdir -p ~/.cache/mediapipe
wget -O ~/.cache/mediapipe/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

To print the versions that are installed on your board:

```
python3 - <<'EOF'
import importlib.metadata as m
for p in ["numpy","pandas","scikit-learn","tqdm","PyYAML","Pillow","transformers","tokenizers","safetensors","huggingface-hub","accelerate","mediapipe","torch","torchvision","jetson-stats","Jetson.GPIO"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "not installed")
EOF
```

**Config in `dms_params.yaml`:** none yet.

## 3. Get the code, the dataset and the models

### Code

Put the project in the folder `~/ros2_dms_code`. Keep this name, because the aliases below use it.

### Dataset (1200 test images)

Download: <https://lobot.cs.queensu.ca/user/group-digi2026-g18/files/phase2-tain-val-test-splits/test_1200.zip?_xsrf=MnwxOjB8MTA6MTc5MjUyMTAwN3w1Ol94c3JmfDE0ODpXbnBXVWpOS2VISnFTbXQxVm5ob1VVeFZhbFJ6ZERGck9YSkxSM0JXVG5abWVFWjFNbEZ5Wm5sbVl6MDZaVGs1WWpJeE1UUTROR001WkRGak1EWmlNekl6WWpZMk1qaGpNMkptWkRrNU5HUmpNMlpsTVdKak5ETmlaR1l3WmpZNFl6Tm1NV00xTUdVM1ltSm1aZz09fDU4NDNhODdiNTIzYmYxYWU0YTBiNGRkZDFiMjZmY2I1MzBmM2I0ZDI4MDBhZjkwOTI5NGVlYmUzYTI4NmIyYTM>

Unpack it so that the folders look like this, with 300 images in each:

```
/home/jetson/test_dataset/NORMAL/
/home/jetson/test_dataset/DROWSY/
/home/jetson/test_dataset/LOOKING_AWAY/
/home/jetson/test_dataset/PHONE_USAGE/
```

Check:

```
for c in NORMAL DROWSY LOOKING_AWAY PHONE_USAGE; do echo $c $(ls /home/jetson/test_dataset/$c | wc -l); done
```

### Models (fine-tuned, LoRA merged)

Put each model in its own folder:

```
/home/jetson/models/SmolVLM-256M-Finetuned/
/home/jetson/models/SmolVLM-500M-Finetuned/
```

Each folder must contain `config.json`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json` and the `*.safetensors` weight files. Check:

```
ls /home/jetson/models/SmolVLM-256M-Finetuned /home/jetson/models/SmolVLM-500M-Finetuned
```

### First build

```
source /opt/ros/foxy/setup.bash
cd ~/ros2_dms_code
colcon build
```

The build must finish with "2 packages finished" (`dms_interfaces` and `dms_ros`).

**Config in `dms_params.yaml`:** if your models are not in `/home/jetson/models`, set `inference_node.models_dir` to the right folder.

## 4. Configure the aliases

Open `~/.bashrc` on the Jetson and add this at the end:

```
source /opt/ros/foxy/setup.bash
source ~/ros2_dms_code/install/setup.bash

alias dms-build='source /opt/ros/foxy/setup.bash && cd ~/ros2_dms_code && colcon build && source install/setup.bash && echo "Build done!"'
alias dms-run='source /opt/ros/foxy/setup.bash && source ~/ros2_dms_code/install/setup.bash && ros2 launch dms_ros dms_launch.py'
alias dms-start='source /opt/ros/foxy/setup.bash && cd ~/ros2_dms_code && colcon build && source install/setup.bash && ros2 launch dms_ros dms_launch.py'
alias dms-free='sudo pkill -9 -f inference_node ; sudo pkill -9 -f camera_node ; sudo pkill -9 -f post_benchmark ; sudo pkill -9 -f ros2 ; sudo pkill -9 -f dms_ros ; sudo systemctl stop gdm3 ; sudo systemctl stop docker ; sudo systemctl stop docker.socket ; sudo systemctl stop snapd ; sudo systemctl stop snapd.socket ; sudo systemctl stop containerd ; sudo pkill -9 -f jtop ; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" && sleep 2 && free -h && echo "Memory freed!"'
```

Quick switches (they change the setting in the source file and in the installed file, so you do not need a rebuild):

```
alias dms-mesh-on='sed -i "s/mesh_enabled: false/mesh_enabled: true/" ~/ros2_dms_code/src/dms_ros/config/dms_params.yaml ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml && grep -n mesh_enabled ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml'
alias dms-mesh-off='sed -i "s/mesh_enabled: true/mesh_enabled: false/" ~/ros2_dms_code/src/dms_ros/config/dms_params.yaml ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml && grep -n mesh_enabled ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml'
alias dms-model-256='sed -i "s#active_model:.*#active_model: \"SmolVLM-256M-Finetuned\"#" ~/ros2_dms_code/src/dms_ros/config/dms_params.yaml ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml && grep -n active_model ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml'
alias dms-model-500='sed -i "s#active_model:.*#active_model: \"SmolVLM-500M-Finetuned\"#" ~/ros2_dms_code/src/dms_ros/config/dms_params.yaml ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml && grep -n active_model ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml'
alias dms-status='grep -n -E "mesh_enabled|active_model|frame_sampling_interval_sec" ~/ros2_dms_code/install/dms_ros/share/dms_ros/config/dms_params.yaml'
```

Then run `source ~/.bashrc`.

| Alias | What it does |
| --- | --- |
| `dms-build` | builds the workspace (needed after every change in the code or the config, see below) |
| `dms-run` | starts the 5 nodes |
| `dms-start` | build and run |
| `dms-free` | stops old nodes and heavy services (desktop, docker, snapd) and frees memory |
| `dms-mesh-on` / `dms-mesh-off` | mesh gate on or off |
| `dms-model-256` / `dms-model-500` | choose the model for the live run |
| `dms-status` | shows the current mesh, model and camera interval |

The build is **not** a symlink install. After you change any file in `src/`, run `dms-build`. The quick-switch aliases above are the only changes that work without a rebuild.

`dms-free` stops the desktop (`gdm3`). Your SSH session keeps working. To get the desktop back: `sudo systemctl start gdm3`.

**Config in `dms_params.yaml`:** none.

## 5. Test the 1200 images without the mesh

This runs the model on every test image with no ROS. It uses the same preprocessing and the same model call as the live system.

**Config used** (`config/dms_params.yaml`, section `inference_node`; the prompts come from `config/prompts.yaml`):

```
inference_node:
  ros__parameters:
    models_dir: "/home/jetson/models"
    active_model: "SmolVLM-256M-Finetuned"   # model used when you give no name on the command line
    max_new_tokens: 12
    longest_edge: 512
    do_sample: false
```

The camera, mesh, alert and benchmark settings are not used here. Keep the values above. The results in this guide were made with them.

Run (one model at a time; each run takes about 2.3 to 2.6 hours). The results in this guide were measured in 5W mode, so use it here:

```
sudo nvpmodel -m 1
dms-free
cd ~/ros2_dms_code/src/dms_ros
nohup python3 validate.py SmolVLM-256M-Finetuned > run_256.log 2>&1 &
tail -f run_256.log
```

Press Ctrl+C to stop `tail` (the test keeps running). When it has finished, run the second model:

```
nohup python3 validate.py SmolVLM-500M-Finetuned > run_500.log 2>&1 &
tail -f run_500.log
```

- It saves every 50 images. If it stops, run the same command again, and it continues.
- To start from zero, delete `validation_results/<model>_dualcrop_predictions.csv` first.

Results in `src/dms_ros/validation_results/`:

| File | Content |
| --- | --- |
| `<model>_dualcrop_predictions.csv` | one row per image: path, true label, prediction, time |
| `<model>_dualcrop_summary.csv` | accuracy, macro-F1, miss rate, false alarm rate, latency |

## 6. Test the 1200 images with the mesh

This does **not** run the model again. For each image, the mesh reads the face (eye openness, head angle, gaze, mouth) and the gate decides: call the model, or skip and answer `NORMAL`. The model answers come from the predictions saved in step 5, so run step 5 first for both models.

**Config used:** none in `dms_params.yaml`. The mesh limits are fixed in `dms_preprocess.py` (`EAR_CLOSED_THRESHOLD = 0.2`, `YAW_AWAY_THRESHOLD_DEG = 30.0` and the any-doubt triage limits). The list of models is `MODELS` at the top of `mesh_gate_eval.py`.

Needs: `mediapipe` and `~/.cache/mediapipe/face_landmarker.task` (see step 2).

```
cd ~/ros2_dms_code/src/dms_ros
python3 mesh_gate_eval.py
```

It takes a few minutes. It prints two gates: **rules only** (skips more, loses accuracy) and **rules + any-doubt triage** (the gate used in the live system). Results in `validation_results/`: `mesh_signals.csv` (mesh values of every image) and `mesh_gate_summary.csv` (one row per model and gate).

## 7. Configure the alarm alias on your laptop

The alert logic runs on the Jetson. When the model says `DROWSY`, the alert node prints `*** ALERT: DRIVER DROWSY ***`, rings the terminal bell and publishes `DROWSY` on `/dms/alert`. The alarm sound plays on your **laptop**, which listens to that topic over SSH. (The Jetson Nano has no speaker.)

**Config in `dms_params.yaml`** (section `alert_node`):

```
alert_node:
  ros__parameters:
    buzzer_pin: 0          # 0 = no buzzer on a GPIO pin
    beep_sec: 2.0
    cooldown_sec: 10.0     # at most one alert every 10 seconds (0 = every DROWSY frame)
    sound_enabled: false   # sound on the Jetson, needs a speaker on the Jetson
    terminal_bell: true    # bell character in the terminal
```

On the laptop you need a Linux system with `ssh` and `paplay` (`sudo apt install pulseaudio-utils`) and the sound file below (it is part of Ubuntu):

```
paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga
```

Add this to the laptop's `~/.bashrc`. Replace `<JETSON_IP>` with the address of your Jetson:

```
dms-alarm() {
  ssh jetson@<JETSON_IP> 'bash -c "source ~/ros2_dms_code/install/setup.bash && PYTHONUNBUFFERED=1 ros2 topic echo /dms/alert std_msgs/msg/String"' | grep --line-buffered DROWSY | while read -r line; do
    timeout 10 bash -c 'while :; do paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga; done'
    pkill -f alarm-clock-elapsed.oga 2>/dev/null
    while read -t 0.2 -r _; do :; done
  done
}
```

Run `source ~/.bashrc`. How to use it:

1. On the Jetson, start the system (step 8 or 9).
2. On the laptop, in a second terminal: `dms-alarm` (enter the SSH password, then leave it open).
3. On every alert the alarm plays for 10 seconds.

Test it without the camera. On the Jetson, in another terminal:

```
ros2 topic pub --once /dms/alert std_msgs/msg/String "{data: DROWSY}"
```

Optional: in Terminator, switch on **Preferences > Profiles > General > Terminal bell > Audible beep** to hear the terminal bell of the Jetson terminal as well.

## 8. Run live, both models, without the mesh

The live sequence is the same each time:

```
dms-build
dms-free
sudo nvpmodel -m 0
sudo jetson_clocks
export CUDA_LAUNCH_BLOCKING=1
dms-run
```

Set the config **before** you start the sequence. Edit `~/ros2_dms_code/src/dms_ros/config/dms_params.yaml` (`nano`), or use the quick aliases. `dms-build` at the start of the sequence installs the new config.

| Setting | SmolVLM-256M | SmolVLM-500M |
| --- | --- | --- |
| `inference_node.active_model` | `"SmolVLM-256M-Finetuned"` | `"SmolVLM-500M-Finetuned"` |
| `preprocess_node.mesh_enabled` | `false` | `false` |
| `camera_node.frame_sampling_interval_sec` | `9.0` | `10.0` |

The camera interval is a start value. Set it a little above the model time that the log prints (`... ms`), so the camera does not make frames that nobody uses. Keep the other values as they are: `max_new_tokens: 12`, `longest_edge: 512`, `do_sample: false`, `queue_size: 1`.

**Run the 256M model:**

```
dms-model-256
dms-mesh-off
dms-status
```

then the live sequence above.

**Run the 500M model:** stop the first run with Ctrl+C, set `frame_sampling_interval_sec: 10.0` in the yaml, then:

```
dms-model-500
dms-mesh-off
dms-status
```

and the live sequence again.

What you see:

1. Start-up lines, then `Preprocess node ready. Mesh OFF` and `Model loaded` (about 1 to 2 minutes).
2. The first model call is slow (up to 1 to 2 minutes). This is the warm-up. After it, one prediction takes about 7 to 10 seconds:
   ```
   Frame #37 -> NORMAL (9160 ms, raw="NORMAL")
   ```
3. On `DROWSY`: `*** ALERT: DRIVER DROWSY ***`, and the alarm on your laptop if `dms-alarm` is running.

The model needs more time than the camera interval, so you see a prediction for about every second frame. This is normal: the queue keeps only the newest frame.

Stop with Ctrl+C. The benchmark is in `/home/jetson/dms_benchmark.csv`. To get the desktop back: `sudo systemctl start gdm3`.

Debug images: to save the full frame, the face crop and the raw image of every frame, set `preprocess_node.debug_dir: "/home/jetson/live_debug"`. Leave it empty (`""`) to switch it off. The file name tells you if the face detector found a face (`faceYES` / `faceNO`).

## 9. Run live, both models, with the mesh

Same sequence as step 8. Only the config changes:

| Setting | SmolVLM-256M | SmolVLM-500M |
| --- | --- | --- |
| `inference_node.active_model` | `"SmolVLM-256M-Finetuned"` | `"SmolVLM-500M-Finetuned"` |
| `preprocess_node.mesh_enabled` | `true` | `true` |
| `preprocess_node.scheduled_interval_sec` | `10.0` | `10.0` |
| `preprocess_node.face_landmarker_path` | `""` | `""` |
| `camera_node.frame_sampling_interval_sec` | `10.0` | `11.0` |

An empty `face_landmarker_path` means `~/.cache/mediapipe/face_landmarker.task`. `scheduled_interval_sec` means the model is called at least that often, even if the mesh says the driver looks normal.

```
dms-model-256        # or dms-model-500
dms-mesh-on
dms-status
```

then the live sequence from step 8.

What you see in addition:

1. `Mesh gate: ear_closed=0.2, yaw_away=30.0, scheduled every 10.0s` and `Preprocess node ready. Mesh ON`.
2. Each prediction line ends with the reason of the gate:
   ```
   Frame #41 -> DROWSY (9826 ms, raw="DROWSY", mesh: mesh_triage)
   ```
3. A skipped frame is not sent to the model:
   ```
   Frame #42 -> NORMAL (mesh skip: confidently_normal, no model call)
   ```

| Gate reason | Meaning |
| --- | --- |
| `uncertain_tracking` | the mesh found no face, the model is called |
| `risk_cue` | eyes closed (EAR < 0.2) or head turned more than 30 degrees, the model is called |
| `mesh_triage` | any doubt (eyes not fully open, head turned or tilted, gaze off-centre, mouth open), the model is called |
| `poor_frame_quality` | blurry frame, the model is called |
| `scheduled_check` | the model has not been called for `scheduled_interval_sec`, it is called now |
| `confidently_normal` | everything looks normal, the model is skipped and the answer is `NORMAL` |

The gate is careful: on the 1200 test images it skips only 2.8% of the frames. In a live setup where the head is not facing the camera straight, almost all frames go to the model. If the mesh cannot start (for example `mediapipe` or the model file is missing), the node logs `Mesh could not start` and runs with the mesh off.

## 10. Settings in `config/dms_params.yaml`

This is the default file of the project:

```yaml
# ================================================================
# DMS Configuration File
# Change values here, rebuild (colcon build), then run.
# ================================================================

# ── Camera Node ───────────────────────────────────────────────────
# source: "camera" = live camera, "replay" = send the images of replay_dir
# one by one and wait for each result (use this to test without a driver).
camera_node:
  ros__parameters:
    source: "replay"
    camera_index: 0
    width: 640
    height: 480
    frame_sampling_interval_sec: 20.0
    queue_size: 1
    replay_dir: "/home/jetson/test_dataset"
    replay_limit_per_class: 0
    replay_timeout_sec: 300.0

# ── Preprocess Node ───────────────────────────────────────────────
# mesh_enabled: false = the model looks at every frame.
# mesh_enabled: true  = the face mesh decides which frames can skip the model.
# Calibrate the mesh limits with:  ros2 run dms_ros mesh_calibrate
# A limit of 0.0 for the optional triage limits switches that limit off.
preprocess_node:
  ros__parameters:
    queue_size: 1
    mesh_enabled: true
    scheduled_interval_sec: 60.0
    face_landmarker_path: "/home/jetson/.cache/mediapipe/face_landmarker.task"
    debug_dir: "/home/jetson/live_debug"
    
# ── Inference Node ────────────────────────────────────────────────
# The prompt for the active model is read from prompts.yaml (same folder).
inference_node:
  ros__parameters:
    models_dir: "/home/jetson/models"
    active_model: "SmolVLM-256M-Finetuned"
    max_new_tokens: 12
    do_sample: false
    longest_edge: 512
    queue_size: 1

# ── Post Benchmark Node ───────────────────────────────────────────
# Every run writes its own files into output_dir:
#   <date>_<time>_<model>_q<queue>_mesh-<on|off>_<source>_summary.csv
#   <date>_<time>_<model>_q<queue>_mesh-<on|off>_<source>_frames.csv
post_benchmark_node:
  ros__parameters:
    output_dir: "/home/jetson/dms_benchmarks"
    flush_interval_sec: 15.0
    monitor_interval_sec: 1.0
```
