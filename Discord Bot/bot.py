
# This file runs a script for a discord application

import io
import logging
import os
import sys
from typing import List, Tuple

import discord
from dotenv import dotenv_values
from PIL import Image, UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
NUMSCAN_DIR = os.path.join(PROJECT_DIR, "Numscan")
if NUMSCAN_DIR not in sys.path:
    sys.path.insert(0, NUMSCAN_DIR)

from network import nn
import predict

DOTENV_CANDIDATES = [
    os.path.join(PROJECT_DIR, ".env"),
    os.path.join(BASE_DIR, ".env"),
]


def load_env_file(dotenv_path: str) -> None:
    with open(dotenv_path, "r", encoding="utf-8-sig") as handle:
        values = dotenv_values(stream=handle)
    for key, value in values.items():
        if value is None:
            continue
        os.environ[key] = value


for dotenv_path in DOTENV_CANDIDATES:
    if os.path.exists(dotenv_path):
        load_env_file(dotenv_path)

ENV_MODEL_PATH = os.getenv("MNIST_MODEL_PATH")
MODEL_CANDIDATES = [
    ENV_MODEL_PATH,
    os.path.join(NUMSCAN_DIR, "Models", "after.pickle"),
    os.path.join(PROJECT_DIR, "Numscan 2", "Models", "model.pkl"),
]
TARGET_CHANNEL = os.getenv("MNIST_CHANNEL_NAME", "numscan")
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
EXPECTED_SIZE = (28, 28)

try:
    RESAMPLE_MODE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_MODE = Image.ANTIALIAS


def load_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    nn.load_from_pickle(model_path)
    logging.info("Loaded model parameters from %s", model_path)


def resolve_model_path(candidates: List[str]) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    missing_list = ", ".join(path for path in candidates if path)
    raise FileNotFoundError(f"Model file not found. Checked: {missing_list}")


def preprocess_image(image: Image.Image) -> List[float]:
    grayscale = image.convert("L")
    resized = grayscale.resize(EXPECTED_SIZE, RESAMPLE_MODE)
    pixels = [pixel / 255.0 for pixel in resized.getdata()]
    if len(pixels) != EXPECTED_SIZE[0] * EXPECTED_SIZE[1]:
        raise ValueError("Unexpected pixel count after preprocessing.")
    return pixels


def preprocess_bytes(data: bytes) -> List[float]:
    with Image.open(io.BytesIO(data)) as img:
        return preprocess_image(img)


def predict_digit(vector: List[float]) -> Tuple[int, float]:
    _, layer_outputs = predict.exec(vector)
    logits = layer_outputs[-1]
    probabilities = predict.soft_max(logits)
    digit = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
    confidence = probabilities[digit]
    return digit, confidence


class NumscanClient(discord.Client):
    def __init__(self, *, target_channel: str, **options):
        super().__init__(**options)
        self.target_channel = target_channel.lower()

    async def on_ready(self) -> None:
        if self.user:
            logging.info("Connected to Discord as %s (id=%s)", self.user, self.user.id)
        else:
            logging.info("Connected to Discord.")

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        channel_name = getattr(message.channel, "name", "").lower()
        if channel_name != self.target_channel:
            return
        relevant_attachments = [
            attachment
            for attachment in message.attachments
            if attachment.filename and attachment.filename.lower().endswith(ALLOWED_EXTENSIONS)
        ]
        if not relevant_attachments:
            return
        responses = []
        for attachment in relevant_attachments:
            try:
                logging.info("Processing attachment %s from %s", attachment.filename, message.author)
                payload = await attachment.read()
                pixels = preprocess_bytes(payload)
                digit, confidence = predict_digit(pixels)
                responses.append(
                    f"{attachment.filename}: {digit} ({confidence * 100:.1f}% confidence)"
                )
            except (UnidentifiedImageError, OSError):
                responses.append(f"{attachment.filename}: could not read image data.")
            except Exception as exc:
                logging.exception("Prediction failed for %s", attachment.filename)
                responses.append(f"{attachment.filename}: prediction failed ({exc}).")
        if responses:
            reply_body = "Predictions:\n" + "\n".join(f"- {line}" for line in responses)
            await message.reply(reply_body, mention_author=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if ENV_MODEL_PATH and not os.path.exists(ENV_MODEL_PATH):
        logging.warning("MNIST_MODEL_PATH is set but missing: %s", ENV_MODEL_PATH)
    model_path = resolve_model_path(MODEL_CANDIDATES)
    load_model(model_path)
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("Please set the DISCORD_BOT_TOKEN environment variable.")
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    client = NumscanClient(intents=intents, target_channel=TARGET_CHANNEL)
    client.run(token)


if __name__ == "__main__":
    main()
