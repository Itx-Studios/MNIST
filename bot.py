import io
import logging
import os
from typing import List, Tuple

import discord
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from network import nn
import predict

load_dotenv()

MODEL_PATH = os.getenv("MNIST_MODEL_PATH", "after.pickle")
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
    load_model(MODEL_PATH)
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
