This project classifies handwritten digits (0-9) 


<div align="center">
  <img src="https://github.com/user-attachments/assets/9409c21d-ad0d-413a-a351-006e9c76ca79" style="width:20%;max-width:250px" />
  <img src="https://github.com/user-attachments/assets/7433cda6-e251-40bb-887a-631b17b1734a" style="width:20%;max-width:250px" />
  <img src="https://github.com/user-attachments/assets/c9bcf798-de87-4402-b117-459298307bb8" style="width:20%;max-width:250px" />
  <img src="https://github.com/user-attachments/assets/9f6cfc45-7ccc-4c36-80a1-b437ba23e65e" style="width:20%;max-width:250px" />
</div>

# ITX Numscan (MNIST)

ITX Numscan is a collection of MNIST digit classifiers and tools. It contains a self-written neural network (Numscan) mostly for educational purpose, a Convolutional neural network version using Tensorflow (Numscan 2), and a Discord bot for Numscan integration in Discord Servers. The Goal of the Project is to learn how neural networks operate and integrating it into Web or Discord.



## Purpose

- To understand the Math behind neural networks
- To provide a accurate classifier
- To integrate digit classification in tools or elsewhere

## Project Structure

- `Numscan/` - Folder for the self-made neural network based classifier 
- `Numscan 2/` - Folder for the newer, smarter classifier with Tensorflow 
- `Discord Bot/` - Folder for Discord integration
- `editor.py` - Desktop testing tool for single samples, either loaded or self-drawn
- `requirements.txt` - All Python requirements

### Set up

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

# 1. Numscan (Dense)

Train the model with the images in the `Data`:

```bash
python "Numscan/Scripts/Train/training.py"
```

Test the models overall performance:

```bash
python "Numscan/Scripts/Test/test.py"
```

Test the model yourself:

```bash
python "editor.py"
```

# 2. Numscan 2 (CNN)

Train a model with the samples of the MNIST Dataset or load an already exsisting one:

```bash
python "Numscan 2/model.py"
```

Test the models performance:

```bash
python "Numscan 2/test.py"
```

Test the model yourself:

```bash
python "editor.py"
```

# 3. Discord Bot

The Discord Application reacts to uploaded images in the correct channel. The application needs join the server for it to work!

Create `Discord Bot/.env` with Token, Channel name and Model path:

```
DISCORD_BOT_TOKEN=your-token-here
MNIST_CHANNEL_NAME=numscan
MNIST_MODEL_PATH=Numscan/Models/after.pickle
```

Run the bot:

```bash
python "Discord Bot/bot.py"
```
