# GPT-From-Scratch: Implementing My Own GPT-Like Model

## 🚀 Project Overview

**GPT-From-Scratch** is my custom implementation of a Decoder-Only Transformer (GPT) architecture, trained end-to-end on a classic childhood story: *Winnie-the-Pooh* 🧸🍯. Built entirely from scratch using PyTorch, my project walks through the full lifecycle of modern generative language models: from building attention mechanisms manually to generating character-level text.

**What I Offer?**
- ✍️ Custom-built transformer architecture and **self-attention** mechanisms (no shortcuts or pretrained libraries)
- ⚙️ Clean separation of logic via a modular `Trainer` class and reusable generation pipeline
- 🧠 Tokenized at the **character-level**, enabling creative and flexible text generation

> 🔗 **[Try it out in Google Colab](https://colab.research.google.com/drive/1pHiw6OKHFPuaUIHw2aJkLGNz1k-cHXLt?usp=sharing)** - generate novel text with your own prompts and settings.

Whether you're technical, non-technical, or just someone nostalgic about storybooks, this repo aims to teach, inspire, and demonstrate the power of transformer models in an intuitive and accessible way.

## ✨ Motivation

As a kid, *Winnie-the-Pooh* wasn't just a story to me, it was my childhood and a world of stories. Recreating that world using artificial intelligence felt like the perfect blend of nostalgia and innovation.

This project began as a challenge to build my own GPT (Generative Pretrained Transformer) **entirely from scratch**. I wanted to understand the mechanics of self-attention, positional embeddings, and autoregressive generation by implementing every component manually.

Beyond technical curiosity, this was also about **having fun** with machine learning. By training on the first 6 chapters of *Winnie-the-Pooh*, my model captures the whimsical, rhythmic tone of A. A. Milne’s writing.

> 🧸 Building something personal made the entire learning process more joyful and reinforced that machine learning can be both rigorous *and* fun.

## 🛠 Features

This project was built completely from scratch using **PyTorch**. It is a minimal, educational implementation of a GPT-like model. Below are some of the key features and components:

### ✅ Core Functionality

- **Character-level Tokenization**  
  Each character is treated as a token, allowing for greater flexibility and creativity in generation.

- **Custom GPT Architecture**  
  Built entirely from scratch, including:
  - Multi-head self-attention
  - Feedforward layers
  - Residual connections and layer normalization

- **Manual Training Loop (GPTTrainer Class)**  
  Includes a reusable and modular training loop with support for:
  - Checkpointing
  - Custom batch sizes
  - Optional validation split
  - Plotting of training/validation loss

- **Text Generation Pipeline**  
  Supports prompt-based generation with:
  - Adjustable `max_length`
  - Configurable temperature for sampling randomness
  - Autoregressive one-token-at-a-time prediction

### 📦 Configurable Training

- Fully customizable via a `TrainingConfig` class:
  - Batch size, learning rate, context length, model dimensions
  - CPU/GPU support
  - Save paths for checkpoints, vocab, and final model

### 📈 Visualization

- Automatic loss tracking and plot generation after training
- Clean training summaries saved to files

### 🌐 Colab-Ready Deployment

- Interactive Colab notebook included
- Users can load the trained model and generate new text live with adjustable settings

This makes the project both a **learning tool** and a **demo platform**. 

## 🧠 Model Architecture

This GPT-like model was designed to be lightweight enough to train on my **laptop CPU**, while still demonstrating the core principles of transformer-based language models.

### 🧩 Key Design Choices

| Component           | Description                                      |
|--------------------|--------------------------------------------------|
| **Tokenization**   | Character-level (1 token = 1 character)          |
| **Context Length** | 64 tokens                                        |
| **Embedding Size** | 128 dimensions                                   |
| **Transformer Blocks** | 4 (each with self-attention + MLP)         |
| **Attention Heads**| 4 (multi-head self-attention)                    |
| **Training Epochs**| 60                                               |
| **Device**         | CPU (no GPU used)                                |

### 🔧 Modules Built From Scratch

- **Multi-Head Self-Attention:** Includes custom masking and projection
- **Feedforward Block:** Feedforward Neural Network + ReLU activation 
- **Residual Connections + LayerNorm:** Stabilize training across blocks
- **Autoregressive Output Head:** Predicts next character based on context

### 🏋️‍♂️ Training Details

This project was trained entirely on my **CPU**, using only the first **6 chapters of _Winnie-the-Pooh_**. The aim was not only to build a working GPT model from scratch but to **generate rich, stylistically consistent text** from a story that’s warm, whimsical, and character-driven.

> Note: My model was trained using all available data (no validation split), and checkpoints were saved periodically to allow recovery in case of interruption.

📊 You can view the final **loss curve** and **training summary** here:  
👉 [`training_outputs/`](https://github.com/Akhan521/GPT-From-Scratch/tree/main/training_outputs)

### ✨ Results & Sample Outputs

After training for 60 epochs on just the **first six chapters of _Winnie-the-Pooh_**, my model learned to generate consistent and surprisingly coherent text, even on a CPU and limited dataset.

Here are a few **sample generations** from my trained model:

<pre>
🐻 Prompt: "Here is Edward Bear, coming"
📝 Output: "Here is Edward Bear, coming downstairs now, bump, bump, on the back of his head, behind Christopher Robin. It is, as far as he..."

🐻 Prompt: "Pooh said"
📝 Output: "Pooh said, "Yes, but it isn't quite a full jar," and he threw it down to Piglet, and Piglet said, "No, it isn't..."

🐻 Prompt: "Christopher Robin"
📝 Output: "Christopher Robin and Pooh went home to breakfast together. "Oh, Bear!" said Christopher Robin. "How I do love you!"
</pre>

These examples demonstrate:

- My model is able to capture the **tone, rhythm, and whimsical charm** of the original story.
- It handles **character dialogue** and **scene-setting** with surprising ability for such a small model.
- Even **custom prompts** result in creative continuations that feel true to the world of the story.

### 🚀 Try It Yourself

There are **two ways** to play around with my trained GPT model:

#### 🔗 1. [Try the Model on Google Colab]([https://colab.research.google.com/drive/1YOUR_COLAB_LINK_HERE](https://colab.research.google.com/drive/1pHiw6OKHFPuaUIHw2aJkLGNz1k-cHXLt?usp=sharing))

> _Recommended for non-technical users or anyone who wants to quickly test the model without setup._

- No installation required
- Run everything from your browser
- Modify prompts, generation length, and temperature

#### 🛠️ 2. Run Locally 

For more details, [click here](https://github.com/Akhan521/GPT-From-Scratch/new/main?filename=README.md#-installation--setup).

### 🧱 Project Structure

The project is cleanly organized to keep the codebase readable, extensible, and easy to navigate. Here's how I structure everything:

```
GPT-FROM-SCRATCH/
├── data/
│   └── training_text.txt            # Raw training text (first 6 chapters of Winnie the Pooh)
├── model/
│   ├── __init__.py
│   └── gpt.py                       # Custom GPT architecture (self-attention, MLP, etc.)
├── notebooks/
│   └── GPT_From_Scratch_Text_Generation.ipynb  # Interactive Colab demo for generation
├── training/
│   ├── __init__.py
│   ├── config.py                    # Centralized config for training (context, lr, etc.)
│   ├── dataset.py                   # PyTorch Dataset with char-level tokenization
│   └── trainer.py                   # GPTTrainer class handling training loop and plots
├── training_outputs/
│   ├── Training_Progress.png        # Plot of training + validation loss
│   └── training_summary.txt         # Summary of training run and hyperparameters
├── utils/
│   ├── __init__.py
│   └── text_generation.py           # Logic for text generation
├── vocab/
│   └── vocab.pkl                    # Serialized vocab (char-to-int, int-to-char)
├── weights/                         # Final trained model + periodic checkpoints
├── .gitignore
├── generate.py                      # CLI for generating text with trained model
├── LICENSE
└── train.py                         # Main script to train the model from scratch
```

## 📦 Installation & Setup

To get started with this project locally, follow the steps below.

### 🔧 Requirements

This project was developed using **Python 3.10** and the following core libraries:

- [PyTorch](https://pytorch.org/) (2.7.1)
- matplotlib
- numpy
- and others (see `requirements.txt`)

> ✅ Although available, no GPU is required.

### 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Akhan521/GPT-From-Scratch.git
cd GPT-From-Scratch
pip install -r requirements.txt
```

> 📁 Make sure your folder structure looks like this:
> - `data/`: contains the training text.
> - `model/`: GPT model implementation.
> - `training/`: dataset, trainer, config.
> - `utils/`: text generation logic.
> - `training_outputs/`: training plots + summary.
> - `weights/`: saved model checkpoint.
> - `notebooks/`: interactive Colab demo.

---

### 🧪 Quick Test (Text Generation)

To test the trained model locally after installing:

```bash
python generate.py
```

This will load the trained model from `weights/final_model.pt` and generate text using the stored vocabulary.

## 🧠 Reflections & Future Work

Building this GPT model from scratch was both a technical deep dive and a nostalgic journey. It pushed me to internalize the inner workings of self-attention, transformer blocks, and the training loop, not just at a conceptual level, but at the implementation level.

### 💡 What I Learned

- How to implement a **Decoder-Only Transformer** without relying on pre-built transformer libraries.
- The importance of proper **tokenization**, training checkpoints, and text generation loops.
- How temperature and context length influence the creativity and coherence of generated text.

### 🔮 What's Next?

This project is just the beginning. Some next steps I'd love to explore:

- **Scaling up**: Training on larger datasets (e.g. full books or datasets).
- **Sampling strategies**: Implement top-k sampling for better generation quality.
- **Interactive UI**: Build a simple web or terminal-based app for live prompt input and generation.
- **Model Evaluation**: Quantify performance using validation.

If you've made it this far into my `README`, thank you for reading and giving me your time!  
I'm always open to feedback, collaboration, and discussion.

🔗 Connect with me:
[GitHub](https://github.com/Akhan521) + [LinkedIn](https://www.linkedin.com/in/aamir-khan-bb83b8235/)

<br>
⭐ If you liked this project, consider starring the repo!




 





