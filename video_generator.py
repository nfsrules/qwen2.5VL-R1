import pygame
import os
import random
import imageio
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
import time
import numpy as np
import cv2
from openai import OpenAI
from tqdm import tqdm


class SyntheticDatasetLoader:
    PROMPT_FORMAT = """
        I will provide you with an image, an original question, and its answer related to the image. 
        Your task is to generate a step-by-step Chain-of-Thought (CoT) reasoning answer with numerical or mathematical expressions where applicable.
        The CoT reasoning process should start with a goal aknowledging expressions like: "The goal is", "So, the objective is". 
        Then, use problem solving expressions like: "Let me think this matter step by step", "let me think", "a way to solve it is", "oh, I see it", "mmm, interesting", "this is probably correct" or other natural language expressions.

        Input Format:
            Original Question: {original_question}
            Original Answer: {original_answer}

        Strict Output Format:
            <think>Your generated CoT reasoning process</think>
            <answer>easy to verify answer (same format than the Original Answer)</answer>

        Important: **Do not include the Original Answer** in the thinking process, only in the final part to avoid leaking the answer.
    """

    def __init__(self, output_dir="dataset", frame_size=64, video_length=30, augment_prob=0.1, augment_types=None):
        base_path = Path(__file__).parent.resolve()
        self.output_dir = (base_path / output_dir).resolve()
        self.video_dir = self.output_dir / "videos"
        self.meta_file = self.output_dir / "metadata.json"
        self.screen_width = self.screen_height = frame_size
        self.video_length = video_length
        self.classes = ["left_to_right", "right_to_left", "falling", "ascending"]
        self.option_labels = {
            "left_to_right": "(A) Left to Right",
            "right_to_left": "(B) Right to Left",
            "falling": "(C) Falling Down",
            "ascending": "(D) Ascending",
        }

        try:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key) if api_key else None
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            self.client = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.augment_prob = augment_prob
        self.augment_types = augment_types or []
        print(f"Dataset will be saved to: {self.output_dir}")
        print(f"Augmentations: {self.augment_types if self.augment_prob > 0 else 'None'}")

    def generate_dataset(self, num_samples=10, cot=False, seed=42, split=0.8):
        if cot and not self.client:
            print("OPENAI_API_KEY not found or OpenAI client unavailable. Falling back to non-CoT generation.")
            cot = False

        print(f"Starting dataset generation ({num_samples} samples, CoT: {cot})...")

        random.seed(seed)
        all_metadata = []

        samples_per_class = num_samples // len(self.classes)
        extra = num_samples % len(self.classes)
        class_list = self.classes * samples_per_class + random.sample(self.classes, extra)
        random.shuffle(class_list)

        num_augmented_videos = int(len(class_list) * self.augment_prob)
        augment_indices = set(random.sample(range(len(class_list)), num_augmented_videos))

        for idx, motion_type in tqdm(enumerate(class_list), total=len(class_list), desc="Generating samples"):
            video_filename = f"{idx:03d}.mp4"
            video_path = self.video_dir / video_filename
            apply_augmentation = idx in augment_indices
            self._generate_video(motion_type, video_path, apply_augmentation)

            question = "<video>\nIn which direction is the ball moving?\nOptions:\n(A) Left to Right\n(B) Right to Left\n(C) Falling Down\n(D) Ascending"
            answer = self.option_labels[motion_type]

            if cot:
                composite_image = self._create_motion_composite(video_path)
                cot_response = self._generate_cot_response(composite_image, question, answer)
                full_answer = cot_response.strip() if cot_response else f"Answer: {answer}"
            else:
                full_answer = f"Answer: {answer}"

            conversations = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": full_answer},
            ]

            all_metadata.append({"video": video_filename, "conversations": conversations})

        split_idx = int(len(all_metadata) * split)
        train_data = all_metadata[:split_idx]
        val_data = all_metadata[split_idx:]

        with open(self.output_dir / "train.json", "w") as f:
            json.dump(train_data, f, indent=2)

        with open(self.output_dir / "val.json", "w") as f:
            json.dump(val_data, f, indent=2)

        print(f"Dataset split saved: {len(train_data)} train / {len(val_data)} val")
        print(f"Video files saved in: {self.video_dir.resolve()}")

    def _generate_video(self, motion_type, save_path, apply_augmentation=False):
        pygame.init()
        screen = pygame.Surface((self.screen_width, self.screen_height))

        black = (0, 0, 0)
        white = (255, 255, 255)
        radius = 20

        if motion_type == "left_to_right":
            x, y = -radius, self.screen_height // 2
            dx, dy = 5, 0
        elif motion_type == "right_to_left":
            x, y = self.screen_width + radius, self.screen_height // 2
            dx, dy = -5, 0
        elif motion_type == "falling":
            x, y = self.screen_width // 2, -radius
            dx, dy = 0, 5
        elif motion_type == "ascending":
            x, y = self.screen_width // 2, self.screen_height + radius
            dx, dy = 0, -5

        frames = []
        for _ in range(self.video_length):
            x += dx
            y += dy
            screen.fill(black)
            pygame.draw.circle(screen, white, (int(x), int(y)), radius)
            frame = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
            frames.append(frame)

        if apply_augmentation:
            frames = self._opencv_augment(frames)

        frames = [Image.fromarray(f.astype("uint8")) for f in frames]
        imageio.mimsave(save_path, frames, fps=30)
        pygame.quit()

    def _opencv_augment(self, frames):
        augmented = []
        for frame in frames:
            img = frame.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if "blur" in self.augment_types:
                ksize = random.choice([3, 5, 7])
                kernel = np.zeros((ksize, ksize))
                kernel[ksize // 2] = np.ones(ksize)
                kernel = kernel / ksize
                img = cv2.filter2D(img, -1, kernel)

            if "crop" in self.augment_types:
                h, w = img.shape[:2]
                crop_frac = random.uniform(0.8, 1.0)
                new_h, new_w = int(h * crop_frac), int(w * crop_frac)
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                img = img[top:top+new_h, left:left+new_w]
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented.append(img)

        return augmented

    def _create_motion_composite(self, video_path):
        reader = imageio.get_reader(video_path)
        num_frames = reader.count_frames()
        sample_indices = [int(i * (num_frames - 1) / 3) for i in range(4)]

        frames = [Image.fromarray(reader.get_data(i)).convert("RGBA") for i in sample_indices]
        base = frames[0].copy()
        for f in frames[1:]:
            base = Image.blend(base, f, alpha=0.5)
        return base.convert("RGB")

    def _generate_cot_response(self, image: Image.Image, question: str, answer: str, max_retries=5):
        def image_to_base64(img: Image.Image):
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

        prompt = self.PROMPT_FORMAT.format(original_question=question, original_answer=answer)
        data_url = image_to_base64(image)

        messages = [
            {"role": "system", "content": "You are an expert to analyze the image and provide a Chain of Thought CoT answer to the user."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]},
        ]

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2**attempt + random.uniform(0, 1))

        print("GPT failed after retries.")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic video dataset for Qwen2.5-VL fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="qwen-vl-finetune/qwenvl/data/synthetic_datasets/my_ball_dataset", help="Where to save videos and metadata.")
    parser.add_argument("--num_samples", type=int, default=20, help="How many video samples to generate.")
    parser.add_argument("--cot", action="store_true", help="Whether to use Chain-of-Thought (CoT) reasoning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio (default: 0.8 for 80% train).")
    parser.add_argument("--frame_size", type=int, default=64, help="Width/height of video frames.")
    parser.add_argument("--video_length", type=int, default=30, help="Number of frames per video.")
    parser.add_argument("--augment_prob", type=float, default=0.1, help="Fraction of videos to augment (0.0 to 1.0). Default: 0.1")
    parser.add_argument("--augment", type=str, default="", help="Comma-separated list of augmentations: blur,crop")

    args = parser.parse_args()

    augment_types = [a.strip() for a in args.augment.split(",") if a.strip()] if args.augment else []

    loader = SyntheticDatasetLoader(
        output_dir=args.output_dir,
        frame_size=args.frame_size,
        video_length=args.video_length,
        augment_prob=args.augment_prob,
        augment_types=augment_types,
    )

    loader.generate_dataset(
        num_samples=args.num_samples,
        cot=args.cot,
        seed=args.seed,
        split=args.split,
    )
