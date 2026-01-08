# ITX Numscan (MNIST)

ITX Numscan is a small collection of MNIST digit classifiers and demo tools. It contains a custom dense neural network implementation (Numscan), a TensorFlow CNN implementation (Numscan 2), and a Discord bot for automated image predictions.

## Purpose

- Learn and compare a handcrafted dense network and a CNN on MNIST.
- Provide simple training and testing scripts for both models.
- Offer GUI tools and a Discord bot to run predictions on user-provided images.

## Project Structure

- `Numscan/` - Custom dense network implementation with training/testing scripts and the local MNIST PNG dataset.
- `Numscan 2/` - TensorFlow CNN model, training/evaluation script, and a GUI tester.
- `Discord Bot/` - Discord bot that reads image uploads and responds with predictions.
- `editor.py` - Desktop GUI to draw or load digits and predict with either model.
- `requirements.txt` - Python dependencies for all tools.

### Install Requirements

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Numscan (Dense Network)

Train the model (uses the PNG dataset in `Numscan/Data/mnist-png`):

```bash
python "Numscan/Scripts/Train/training.py"
```

Test the model after training:

```bash
python "Numscan/Scripts/Test/test.py"
```

The trained weights are stored at `Numscan/Models/after.pickle`.

### Numscan 2 (CNN)

Train or load the CNN model and evaluate it:

```bash
python "Numscan 2/model.py"
```

This script downloads MNIST automatically the first time and saves weights to `Numscan 2/Models/model.pkl`.

Launch the GUI tester for drawing, demo samples, and uploads:

```bash
python "Numscan 2/test.py"
```

### Editor GUI (Unified Tester)

Use the desktop editor to draw or load digits and run predictions with Numscan or Numscan 2:

```bash
python editor.py
```

Make sure the corresponding model files exist:

- Numscan: `Numscan/Models/after.pickle`
- Numscan 2: `Numscan 2/Models/model.pkl`

### Discord Bot

The bot listens in a configured channel, reads image attachments, and responds with predicted digits. It expects a Numscan-compatible pickle model.

Create `Discord Bot/.env` with at least:

```
DISCORD_BOT_TOKEN=your-token-here
MNIST_CHANNEL_NAME=numscan
MNIST_MODEL_PATH=Numscan/Models/after.pickle
```

Run the bot:

```bash
python "Discord Bot/bot.py"
```
