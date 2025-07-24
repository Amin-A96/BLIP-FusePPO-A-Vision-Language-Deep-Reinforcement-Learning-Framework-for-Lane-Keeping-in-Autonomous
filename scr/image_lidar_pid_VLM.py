import os
import cv2
import numpy as np
import random
import time
import torch
from torch import nn
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

from vehicle import Driver
from controller import Supervisor
import gym
from gym import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from transformers import BlipProcessor, BlipForConditionalGeneration


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


SEED = 42
set_seed(SEED)


IMAGE_HEIGHT = 300
IMAGE_WIDTH = 700
IMAGE_HEIGHT_cnn = 224
IMAGE_WIDTH_cnn = 244
OUTPUT_VECTOR_SIZE = 10
MAX_STEERING_ANGLE = 0.8
MAX_SPEED = 20.0
SMOOTHING_ALPHA = 0.2
MIN_LINE_LENGTH = 3
MAX_LINE_GAP = 100
lidar_p = 0
t1 = 0
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def preprocess_image(image):
    if image is None or image.size == 0:
        print("Error: Invalid image provided to preprocess_image.")
        return None
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(image_norm, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def calculate_bottom_point(line):
    x1, y1, x2, y2 = line
    return (x1, y1) if y1 > y2 else (x2, y2)

def group_and_select_lines(lines, width):
    center_x = width // 2
    left_cands, right_cands = [], []
    if lines is None:
        return None, None
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2 - x1, y2 - y1)
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) > 100:
            continue
        bx, by = calculate_bottom_point((x1, y1, x2, y2))
        if slope < -0.2 and (x1 < center_x or x2 < center_x or abs(bx - center_x) < width * 0.1):
            left_cands.append(((x1, y1, x2, y2), bx, by, length))
        elif slope > 0.2 and (x1 > center_x or x2 > center_x or abs(bx - center_x) < width * 0.1):
            right_cands.append(((x1, y1, x2, y2), bx, by, length))
    def score(item):
        _, bx, by, length = item
        return (by, -abs(center_x - bx), length)
    left_line = max(left_cands, key=score)[0] if left_cands else None
    right_line = max(right_cands, key=score)[0] if right_cands else None
    return left_line, right_line

def draw_and_smooth(image, left_line, right_line, width, height, alpha=SMOOTHING_ALPHA, prev={'left': None, 'right': None}):
    center_x = width // 2
    bottom_y = height - 1
    cv2.circle(image, (center_x, bottom_y), 5, (0, 0, 255), -1)
    cv2.putText(image, "Car Center", (center_x - 40, bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    raw = {}
    raw['left'] = calculate_bottom_point(left_line) if left_line else (0, height)
    raw['right'] = calculate_bottom_point(right_line) if right_line else (width, height)
    smooth = {}
    for side in ['left', 'right']:
        if prev[side] is None:
            smooth[side] = raw[side]
        else:
            sx = int(alpha * raw[side][0] + (1 - alpha) * prev[side][0])
            sy = int(alpha * raw[side][1] + (1 - alpha) * prev[side][1])
            smooth[side] = (sx, sy)
        prev[side] = smooth[side]
        color = (0, 255, 255) if side == 'left' else (255, 255, 0)
        cv2.circle(image, smooth[side], 5, color, -1)
        cv2.putText(image, side.capitalize(), (smooth[side][0]+5, smooth[side][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    mx = (smooth['left'][0] + smooth['right'][0]) // 2
    my = (smooth['left'][1] + smooth['right'][1]) // 2
    cv2.circle(image, (mx, my), 5, (0, 255, 0), -1)
    cv2.putText(image, "Mid", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
    distance = mx - center_x
    return distance

def draw_hough_lines(image, lines):
    hough_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return hough_image


class EnhancedRoadConditionAnalyzer:
    def __init__(self, device=None, model_name="Salesforce/blip-image-captioning-large", image_size=224, use_sampling=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.image_size = image_size
        self.use_sampling = use_sampling

    def _generate_caption(self, inputs):
        config = {
            "max_length": 300,
            "min_length": 50,
            "num_beams": 5,
            "no_repeat_ngram_size": 2,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **config)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def caption_to_vector(self, caption, max_length=50):
        encoding = self.processor.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        token_vector = encoding["input_ids"].squeeze(0)
        return token_vector

    def vector_to_token_words(self, token_vector):
        token_ids = token_vector.tolist()
        token_words = self.processor.tokenizer.convert_ids_to_tokens(token_ids)
        return token_words

    def analyze_image(self, image_path, max_length=50):
        try:
            original_image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
            inputs = self.processor(
                images=original_image,
                text="Describe the road condition and its surroundings:",
                return_tensors="pt"
            ).to(self.device)
            caption = self._generate_caption(inputs)
            caption_vector = self.caption_to_vector(caption, max_length=max_length)
            token_words = self.vector_to_token_words(caption_vector)
            return caption, caption_vector, token_words, original_image
        except Exception as e:
            return f"Error processing image {image_path}: {str(e)}", None, None, None


class PIDController:
    def __init__(self, Kp=0.15, Ki=0.02, Kd=0.08, max_integral=1.0, max_output=MAX_STEERING_ANGLE):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_integral = max_integral
        self.max_output = max_output
        self.reset()
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, error, dt=1.0):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        output = np.clip(output, -self.max_output, self.max_output)
        self.prev_error = error
        return output


class SplitLaneFollowingEnv(Env):
    supervisor_instance = None
    driver_instance = None

    def __init__(self, car_def_name="MY_ROBOT"):
        super().__init__()
        self.np_random = np.random.RandomState(SEED)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 1, (IMAGE_HEIGHT_cnn, IMAGE_WIDTH_cnn, 3), dtype=np.float32),
            "pid_correction": spaces.Box(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE, (1,), dtype=np.float32),
            "lidar": spaces.Box(0, 1, (180,), dtype=np.float32),
            "token_vector": spaces.Box(0, 30521, (50,), dtype=np.int64)
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        
        self.steering_pid = PIDController(Kp=0.15, Ki=0.02, Kd=0.08)
        self.last_distance_error = 0.0
        if SplitLaneFollowingEnv.supervisor_instance is None:
            SplitLaneFollowingEnv.supervisor_instance = Supervisor()
        self.supervisor = SplitLaneFollowingEnv.supervisor_instance

        if SplitLaneFollowingEnv.driver_instance is None:
            SplitLaneFollowingEnv.driver_instance = Driver()
        self.driver = SplitLaneFollowingEnv.driver_instance

        self.car_def_name = car_def_name
        self.car_node = self.supervisor.getFromDef(self.car_def_name)
        if self.car_node is None:
            print(f"Error: Could not find Automobile with DEF '{self.car_def_name}'")
        try:
            self.camera = self.supervisor.getDevice("camera")
            self.camera.enable(int(self.supervisor.getBasicTimeStep()))
        except Exception:
            print("Warning: No camera device found. Camera code will be skipped.")
            self.camera = None
        try:
            self.lidar = self.supervisor.getDevice("Sick LMS 291")
            self.lidar.enable(int(self.supervisor.getBasicTimeStep()))
            self.lidar.enablePointCloud()
        except Exception:
            print("Warning: No LiDAR device found. LiDAR code will be skipped.")
            self.lidar = None
        
        self.road_condition_analyzer = EnhancedRoadConditionAnalyzer()
        self.blip_update_interval = 10
        self.frame_counter = 0
        self.last_token_vector = np.zeros(50, dtype=np.int64)
        self.last_aug_token_vector = np.zeros(50, dtype=np.int64)
        self.episode_count = 0
        self.start_positions = [
            {"translation": [-36.9526, 119.667, 0.342393], "rotation": [-0.00224826, 0.00228126, 0.999995,1.56]},
            {"translation": [-42.4526, 119.667, 0.342393], "rotation": [-0.00224826, 0.00228126, 0.999995,1.56]},
        ]
        self.start_index = 0
        self.current_start_position = self.start_positions[self.start_index]
        self.blip_cache = {}

        self.lidar_below_2_count = 0
        self.lidar_timestep_counter = 0
        self.lidar_episode_counter = 0
        self.first_lidar_below_2_episode = None
        self.lidar_below_2_episodes = []
        self.phase = 0
        self.last_augmented_transition = None

    def seed(self, seed=None):
            self.np_random = np.random.RandomState(seed)
            return [seed]
        
    def _process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (244, 224))
        return image.astype(np.float32) / 255.0

    def _get_token_vector(self, image):
        cache_key = hash(image.tobytes())
        if cache_key in self.blip_cache:
            return self.blip_cache[cache_key]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.fromarray(image).save(tmp.name)
            caption, vector, _, _ = self.road_condition_analyzer.analyze_image(tmp.name, max_length=50)
        vector_np = vector.numpy()
        if len(self.blip_cache) >= 100:
            self.blip_cache.pop(next(iter(self.blip_cache)))
        self.blip_cache[cache_key] = vector_np
        return vector_np

    def _get_state(self, done=False):
        if self.camera:
            image_data = self.camera.getImage()
            if image_data is not None:
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                bottom_part = image_rgb[IMAGE_HEIGHT - 110:, :, :]
                network_img = cv2.resize(bottom_part, (IMAGE_WIDTH_cnn, IMAGE_HEIGHT_cnn))
                image_normalized = network_img.astype('float32') / 255.0
            else:
                image_normalized = np.zeros((IMAGE_HEIGHT_cnn, IMAGE_WIDTH_cnn, 3), dtype=np.float32)
        else:
            image_normalized = np.zeros((IMAGE_HEIGHT_cnn, IMAGE_WIDTH_cnn, 3), dtype=np.float32)
        
        lidar_data = np.zeros(180, dtype=np.float32)
        if self.lidar:
            lidar_data = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_data[np.isinf(lidar_data)] = 100.0
        
        token_vector = np.zeros(50, dtype=np.int64)
        if self.camera:
            if done or (self.frame_counter % self.blip_update_interval == 0):
                try:
                    pil_image = Image.fromarray(image_rgb) 
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                        temp_filename = tmp_file.name
                        pil_image.save(temp_filename)
                    caption, caption_vector, _, _ = self.road_condition_analyzer.analyze_image(temp_filename, max_length=50)
                    if caption_vector is not None:
                        token_vector = caption_vector.numpy()
                    self.last_token_vector = token_vector
                    os.remove(temp_filename)
                except Exception as e:
                    print("Error generating token vector:", e)
                    token_vector = self.last_token_vector
            else:
                token_vector = self.last_token_vector
        
        distance_error = self.last_distance_error
        dt = self.supervisor.getBasicTimeStep() / 1000.0
        pid_correction = self.steering_pid.compute(distance_error, dt)
        
        return {
            "image": image_normalized,
            "pid_correction": np.array([pid_correction], dtype=np.float32),
            "lidar": lidar_data,
            "token_vector": token_vector
        }

    def _get_augmented_state(self, done=False):
        if self.camera:
            image_data = self.camera.getImage()
            if image_data is not None:
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                aug_rgb = cv2.flip(image_rgb, 1) 
                aug_bottom_part = aug_rgb[IMAGE_HEIGHT - 110:, :, :]
                network_img = cv2.resize(aug_bottom_part, (IMAGE_WIDTH_cnn, IMAGE_HEIGHT_cnn))
                aug_image = network_img.astype('float32') / 255.0
            else:
                aug_image = np.zeros((IMAGE_HEIGHT_cnn, IMAGE_WIDTH_cnn, 3), dtype=np.float32)
        else:
            aug_image = np.zeros((IMAGE_HEIGHT_cnn, IMAGE_WIDTH_cnn, 3), dtype=np.float32)
        
        lidar_data = np.zeros(180, dtype=np.float32)
        if self.lidar:
            lidar_data = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_data[np.isinf(lidar_data)] = 100.0
        aug_lidar = np.flip(lidar_data)
        
        augmented_token = np.zeros(50, dtype=np.int64)
        if self.camera:
            if done or (self.frame_counter % self.blip_update_interval == 0):
                try:
                    aug_rgb = cv2.flip(image_rgb, 1) 
                    pil_aug_image = Image.fromarray(aug_rgb)
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                        temp_aug_filename = tmp_file.name
                        pil_aug_image.save(temp_aug_filename)
                    caption_aug, caption_vector_aug, _, _ = self.road_condition_analyzer.analyze_image(temp_aug_filename, max_length=50)
                    if caption_vector_aug is not None:
                        augmented_token = caption_vector_aug.numpy()
                    self.last_aug_token_vector = augmented_token
                    os.remove(temp_aug_filename)
                except Exception as e:
                    print("Error generating augmented token vector:", e)
                    augmented_token = self.last_aug_token_vector
            else:
                augmented_token = self.last_aug_token_vector
        
        distance_error = -self.last_distance_error  
        dt = self.supervisor.getBasicTimeStep() / 1000.0
        pid_correction = self.steering_pid.compute(distance_error, dt)
        
        return {
            "image": aug_image,
            "pid_correction": np.array([pid_correction], dtype=np.float32),
            "lidar": aug_lidar,
            "token_vector": augmented_token
        }

    def step(self, action):
        global t1
        if self.phase == 0:
            distance_error = self.last_distance_error
            dt = self.supervisor.getBasicTimeStep() / 1000.0
            pid_correction = self.steering_pid.compute(distance_error, dt)
            base_steer = action[0] * MAX_STEERING_ANGLE
            self.driver.setSteeringAngle(base_steer)
            speed = (action[1] + 1.0) * MAX_SPEED / 2
            self.driver.setCruisingSpeed(speed)
            self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
            self.frame_counter += 1

            state = self._get_state()
    
            distance = 0
            if self.camera:
                img_buf = self.camera.getImage()
                if img_buf is not None:
                    img = np.frombuffer(img_buf, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    crop = image_rgb[IMAGE_HEIGHT-110:IMAGE_HEIGHT, :]
                    pre = preprocess_image(crop)
                    lines = cv2.HoughLinesP(pre, 1, np.pi/60, 15, minLineLength=5, maxLineGap=100)
                    left, right = group_and_select_lines(lines, IMAGE_WIDTH)
                    distance = draw_and_smooth(crop.copy(), left, right, IMAGE_WIDTH, 110)
                else:
                    distance = 0
            else:
                distance = 0
    
            lidar_data = np.zeros(180, dtype=np.float32)
            if self.lidar:
                lidar_data = np.array(self.lidar.getRangeImage(), dtype=np.float32)
            lidar_data[np.isinf(lidar_data)] = 100.0
            min_distance = np.min(lidar_data)
            self.last_distance_error = distance
    
            done = False
            if min_distance < 2:
                self.lidar_below_2_count += 1
                if self.first_lidar_below_2_episode is None:
                    self.first_lidar_below_2_episode = self.episode_count
                self.lidar_below_2_episodes.append(self.episode_count)
    
            self.lidar_timestep_counter += 1
    
            if self.lidar_timestep_counter >= 20:
                if self.lidar_below_2_count >= 10:
                    done = True
                    reward = -3 + (self.supervisor.getTime() - t1) / 50
                    print("---------------Lidar reset episode (3 times < 2 in 20 timesteps) -----------------")
                self.lidar_below_2_count = 0
                self.lidar_timestep_counter = 0
    
            if self.first_lidar_below_2_episode is not None:
                if self.episode_count - self.first_lidar_below_2_episode >= 20:
                    if sum(1 for ep in self.lidar_below_2_episodes if ep > self.first_lidar_below_2_episode) >= 2:
                        done = True
                        reward = -3 + (self.supervisor.getTime() - t1) / 50
                        print("---------------Lidar reset episode (3 times < 2 in 20 episodes) -----------------")
                    self.first_lidar_below_2_episode = None
                    self.lidar_below_2_episodes = []
    
            if not done:
                distance_reward = 1 - (abs(distance) / 100)
    
                desired_speed = 20.0
                speed_reward = - (speed - desired_speed) ** 2 / (desired_speed ** 2)
    
                if 4 <= min_distance <= 8:
                    lidar_penalty = -5 * (8 - min_distance) / 4
                elif min_distance < 2.8:
                    lidar_penalty = -10 + 2 * min_distance
                else:
                    lidar_penalty = 0
    
                if 3 <= min_distance <= 4 or 8 <= min_distance <= 10:
                    bonus = 5
                else:
                    bonus = 0
    
                lidar_reward = lidar_penalty + bonus
    
                max_distance = 80.0
                k = 2.5
                center_distance_reward = -k * (abs(distance) / max_distance) ** 2
    
                w_distance = 0.3
                w_speed = 0.2
                w_lidar = 0.3
                w_center = 0.2
    
                reward = (w_distance * distance_reward + 
                          w_speed * speed_reward + 
                          w_lidar * lidar_reward / 10 + 
                          w_center * center_distance_reward)
                reward = np.clip(reward, -1, 1)
    
            if abs(distance) > 85:
                reward = -3 + (self.supervisor.getTime() - t1) / 50
                done = True
                print("---------------camera reset episode -----------------")
    
            token_vector = state["token_vector"]
            if self.camera and (done or (self.frame_counter % self.blip_update_interval == 0)):
                try:
                    pil_image = Image.fromarray(image_rgb)
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                        temp_filename = tmp_file.name
                        pil_image.save(temp_filename)
                    caption, caption_vector, _, _ = self.road_condition_analyzer.analyze_image(temp_filename, max_length=50)
                    if caption_vector is not None:
                        token_vector = caption_vector.numpy()
                    self.last_token_vector = token_vector
                    os.remove(temp_filename)
                except Exception as e:
                    print("Error generating token vector:", e)
                    token_vector = self.last_token_vector
    
            original_state = {
                "image": state["image"],
                "pid_correction": state["pid_correction"],
                "lidar": lidar_data,
                "token_vector": token_vector
            }
    
            augmented_state = self._get_augmented_state(done)
            augmented_action = np.array([-action[0], action[1]])
            self.last_augmented_transition = (augmented_state, reward, done, {"augmented_action": augmented_action})
            self.phase = 1
            return original_state, reward, done, {}
        else:
            self.phase = 0
            return self.last_augmented_transition
    
    def reset(self):
        self.steering_pid.reset()
        self.last_distance_error = 0.0
        self.episode_count += 1
        self.lidar_below_2_count = 0
        self.lidar_timestep_counter = 0
    
        if self.car_node is None:
            blank_state = {
                "image": np.zeros((IMAGE_HEIGHT_cnn, IMAGE_WIDTH_cnn, 3), dtype=np.float32),
                "pid_correction": np.array([0.0], dtype=np.float32),
                "lidar": np.zeros(180, dtype=np.float32),
                "token_vector": np.zeros(50, dtype=np.int64)
            }
            self.phase = 0
            return blank_state
    
        translation_field = self.car_node.getField("translation")
        rotation_field = self.car_node.getField("rotation")
    
        if self.episode_count % 3 == 0:
            self.start_index = (self.start_index + 1) % len(self.start_positions)
            self.current_start_position = self.start_positions[self.start_index]
            print(f"Episode {self.episode_count}: Changing start position to index {self.start_index}: {self.current_start_position}")
    
        translation_field.setSFVec3f(self.current_start_position["translation"])
        rotation_field.setSFRotation(self.current_start_position["rotation"])
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        state_original = self._get_state(done=True)
        self.phase = 0
        return state_original

class CustomCNNWithLiDARAndTokenAugmented(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.spaces["image"].shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            sample_img = torch.as_tensor(observation_space.spaces["image"].sample()[None]).permute(0, 3, 1, 2).float()
            cnn_output_dim = self.cnn(sample_img).shape[1]
        self.lidar_fc = nn.Sequential(
            nn.Linear(180, 64),
            nn.ReLU()
        )
        self.token_embedding = nn.Embedding(30522, 32)
        self.token_fc = nn.Sequential(
            nn.Linear(50 * 32, 64),
            nn.ReLU()
        )
        self.pid_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        branch_feature_dim = cnn_output_dim + 64 + 64 + 16
        self.fc = nn.Sequential(
            nn.Linear(branch_feature_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        img = observations["image"].permute(0, 3, 1, 2)
        cnn_features = self.cnn(img)
        lidar_features = self.lidar_fc(observations["lidar"])
        token = observations["token_vector"].long()
        token_embedded = self.token_embedding(token)
        token_features = self.token_fc(token_embedded.view(token.size(0), -1))
        pid_features = self.pid_fc(observations["pid_correction"])
        combined = torch.cat([cnn_features, lidar_features, token_features, pid_features], dim=1)
        return self.fc(combined)

policy_kwargs = dict(
    features_extractor_class=CustomCNNWithLiDARAndTokenAugmented,
    features_extractor_kwargs=dict(features_dim=128),
)

def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')

class RewardPlottingCallback(BaseCallback):
    def __init__(self, env, model, save_path="reward_plots", plot_interval=1000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.model = model
        self.save_path = os.path.abspath(save_path)
        self.episode_rewards = []
        self.episodes = []
        self.timesteps = []
        self.total_timesteps = 0
        self.plot_interval = plot_interval
        
        self.speeds = []
        self.steerings_with_pid = []
        self.steerings_without_pid = []
        self.distances = []
        self.rewards = []
        self.lidar_sensor = []

        self.current_episode_reward = 0.0
        
        self.rollout_metrics = {"ep_len_mean": [], "ep_rew_mean": []}
        
        self.ppo_timesteps = []
        self.ppo_approx_kl = []
        self.ppo_loss = []
        self.ppo_explained_variance = []
        self.ppo_learning_rate = []
        self.ppo_clip_fraction = []
        self.ppo_entropy_loss = []
        self.ppo_policy_gradient_loss = []
        self.ppo_value_loss = []

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.total_timesteps += 1
        
        speed = self.env.driver.getTargetCruisingSpeed()
        action = self.locals["actions"][-1]
        base_steer = action[0] * 0.8
        dt = 0.1
        pid_correction = self.env.steering_pid.compute(self.env.last_distance_error, dt)
        final_steer = np.clip(base_steer + pid_correction, -0.8, 0.8)
        if self.env.lidar:
            lidar_data = np.array(self.env.lidar.getRangeImage(), dtype=np.float32)
            lidar_data[np.isinf(lidar_data)] = 100.0
            lidar_min = np.min(lidar_data)
        else:
            lidar_min = 100.0
        self.lidar_sensor.append(lidar_min)
        self.steerings_with_pid.append(final_steer)
        self.steerings_without_pid.append(base_steer)
        self.speeds.append(speed)
        distance = self.env.last_distance_error
        self.distances.append(distance)
        self.rewards.append(self.locals["rewards"][-1])

        self.current_episode_reward += self.locals["rewards"][-1]

        if hasattr(self.model, "logger"):
            logger = self.model.logger
            if logger.name_to_value:
                ep_len_mean = logger.name_to_value.get("rollout/ep_len_mean", np.nan)
                ep_rew_mean = logger.name_to_value.get("rollout/ep_rew_mean", np.nan)
                if ep_len_mean is not np.nan:
                    self.rollout_metrics["ep_len_mean"].append(ep_len_mean)
                if ep_rew_mean is not np.nan:
                    self.rollout_metrics["ep_rew_mean"].append(ep_rew_mean)

                if "train/approx_kl" in logger.name_to_value:
                    self.ppo_timesteps.append(self.total_timesteps)
                    self.ppo_approx_kl.append(logger.name_to_value.get("train/approx_kl", np.nan))
                    self.ppo_loss.append(logger.name_to_value.get("train/loss", np.nan))
                    self.ppo_explained_variance.append(logger.name_to_value.get("train/explained_variance", np.nan))
                    self.ppo_learning_rate.append(logger.name_to_value.get("train/learning_rate", np.nan))
                    self.ppo_clip_fraction.append(logger.name_to_value.get("train/clip_fraction", np.nan))
                    self.ppo_entropy_loss.append(logger.name_to_value.get("train/entropy_loss", np.nan))
                    self.ppo_policy_gradient_loss.append(logger.name_to_value.get("train/policy_gradient_loss", np.nan))
                    self.ppo_value_loss.append(logger.name_to_value.get("train/value_loss", np.nan))

        if self.total_timesteps % self.plot_interval == 0:
            self._plot_ppo_metrics()

        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(self.current_episode_reward)
                self.episodes.append(len(self.episode_rewards))
                self.timesteps.append(self.total_timesteps)
                self._plot_metrics()
                self.current_episode_reward = 0.0
        return True

    def _plot_metrics(self):
        fig, axes = plt.subplots(6, 1, figsize=(12, 24))

        axes[0].plot(self.episodes, self.episode_rewards, marker="o", linestyle="-", color="b", label="Cumulative Reward")
        ma_rewards = moving_average(self.episode_rewards[1:], window=50)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].set_title("Cumulative Reward per Episode")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(range(len(self.speeds)), self.speeds, linestyle="-", color="g", label="Speed (km/h)")
        ma_speed = moving_average(self.speeds)
        axes[1].plot(range(len(ma_speed)), ma_speed, linestyle="--", color="black", label="Moving Average")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Speed (km/h)")
        axes[1].set_title("Speed Over Time")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(range(len(self.steerings_without_pid)), self.steerings_without_pid, linestyle="--", color="orange", label="Without PID")
        ma_without = moving_average(self.steerings_without_pid)
        ma_with = moving_average(self.steerings_with_pid)
        axes[2].plot(range(len(ma_without)), ma_without, linestyle="--", color="black", label="MA Without PID")
        axes[2].plot(range(len(ma_with)), ma_with, linestyle="-.", color="gray", label="MA With PID")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Steering Angle (°)")
        axes[2].set_title("Steering Angle Comparison: With PID vs Without PID")
        axes[2].grid(True)
        axes[2].legend()

        axes[3].plot(range(len(self.distances)), self.distances, linestyle="-", color="m", label="Distance from Center")
        ma_distance = moving_average(self.distances)
        axes[3].plot(range(len(ma_distance)), ma_distance, linestyle="--", color="black", label="Moving Average")
        axes[3].set_xlabel("Timestep")
        axes[3].set_ylabel("Distance (m)")
        axes[3].set_title("Distance from Lane Center Over Time")
        axes[3].grid(True)
        axes[3].legend()

        axes[4].plot(range(len(self.rewards)), self.rewards, linestyle="-", color="c", label="Instantaneous Reward")
        ma_inst_reward = moving_average(self.rewards)
        axes[4].plot(range(len(ma_inst_reward)), ma_inst_reward, linestyle="--", color="black", label="Moving Average")
        axes[4].set_xlabel("Timestep")
        axes[4].set_ylabel("Reward")
        axes[4].set_title("Instantaneous Reward Over Time")
        axes[4].grid(True)
        axes[4].legend()
        
        axes[5].plot(range(len(self.lidar_sensor)), self.lidar_sensor, linestyle="-", color="purple", label="LiDAR Min Distance")
        ma_lidar = moving_average(self.lidar_sensor)
        axes[5].plot(range(len(ma_lidar)), ma_lidar, linestyle="--", color="black", label="Moving Average")
        axes[5].set_xlabel("Timestep")
        axes[5].set_ylabel("Distance (m)")
        axes[5].set_title("LiDAR Sensor Min Distance Over Time")
        axes[5].grid(True)
        axes[5].legend()
        
        plt.tight_layout()
        save_file = os.path.join(self.save_path, f"main_metrics_plot_episode_{len(self.episode_rewards)}.png")
        plt.savefig(save_file, dpi=300)
        print(f"Saved main metrics plot to: {save_file}")
        plt.close(fig)

    def _plot_ppo_metrics(self):
        if not self.ppo_timesteps:
            return

        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()

        axes[0].plot(self.ppo_timesteps, self.ppo_approx_kl, linestyle="-", color="b", label="Approx KL")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Approx KL")
        axes[0].set_title("Approx KL Over Time")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(self.ppo_timesteps, self.ppo_loss, linestyle="-", color="g", label="Loss")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss Over Time")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(self.ppo_timesteps, self.ppo_explained_variance, linestyle="-", color="r", label="Explained Variance")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Explained Variance")
        axes[2].set_title("Explained Variance Over Time")
        axes[2].grid(True)
        axes[2].legend()

        axes[3].plot(self.ppo_timesteps, self.ppo_learning_rate, linestyle="-", color="m", label="Learning Rate")
        axes[3].set_xlabel("Timestep")
        axes[3].set_ylabel("Learning Rate")
        axes[3].set_title("Learning Rate Over Time")
        axes[3].grid(True)
        axes[3].legend()

        axes[4].plot(self.ppo_timesteps, self.ppo_clip_fraction, linestyle="-", color="c", label="Clip Fraction")
        axes[4].set_xlabel("Timestep")
        axes[4].set_ylabel("Clip Fraction")
        axes[4].set_title("Clip Fraction Over Time")
        axes[4].grid(True)
        axes[4].legend()

        axes[5].plot(self.ppo_timesteps, self.ppo_entropy_loss, linestyle="-", color="y", label="Entropy Loss")
        axes[5].set_xlabel("Timestep")
        axes[5].set_ylabel("Entropy Loss")
        axes[5].set_title("Entropy Loss Over Time")
        axes[5].grid(True)
        axes[5].legend()

        axes[6].plot(self.ppo_timesteps, self.ppo_policy_gradient_loss, linestyle="-", color="k", label="Policy Gradient Loss")
        axes[6].set_xlabel("Timestep")
        axes[6].set_ylabel("Policy Gradient Loss")
        axes[6].set_title("Policy Gradient Loss Over Time")
        axes[6].grid(True)
        axes[6].legend()

        axes[7].plot(self.ppo_timesteps, self.ppo_value_loss, linestyle="-", color="orange", label="Value Loss")
        axes[7].set_xlabel("Timestep")
        axes[7].set_ylabel("Value Loss")
        axes[7].set_title("Value Loss Over Time")
        axes[7].grid(True)
        axes[7].legend()

        plt.tight_layout()
        save_file = os.path.join(self.save_path, f"ppo_metrics_plot_timestep_{self.total_timesteps}.png")
        plt.savefig(save_file, dpi=300)
        print(f"Saved PPO metrics plot to: {save_file}")
        plt.close(fig)

class HybridPPO(PPO):
    def train(self):
        super().train()



if __name__ == "__main__":
    env = SplitLaneFollowingEnv()
    env.seed(SEED)
    print("Creating new PPO model...")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        verbose=1,
        seed=SEED,
        device="auto"
    )
    reward_plot_callback = RewardPlottingCallback(
        env=env,
        model=model,
        save_path="reward_plots",
        plot_interval=500,
        verbose=1
    )
    print("Training new model...")
    model.learn(total_timesteps=100000, callback=reward_plot_callback)

    print("Saving new model...")
    model.save("hybrid_lane_following_agent")

    print("Training completed.")
    
    
# if __name__ == "__main__":
    # env = SplitLaneFollowingEnv()

    # model_path = "hybrid_lane_following_agent.zip" 
    # print(f"Loading model from {model_path}...")
    # model = PPO.load(model_path, env=env)

    # num_test_episodes = 5
    # for episode in range(1, num_test_episodes + 1):
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0.0
    #     step = 0
    #     print(f"--- Test Episode {episode} ---")
    #     while not done:
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
    #         total_reward += reward
    #         step += 1

    #     print(f"Episode {episode} finished in {step} timesteps, total reward = {total_reward:.2f}")

    # print("Testing completed.")