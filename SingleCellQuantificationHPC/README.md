# Fungal Single Cell Quantification - Manual Correction Tool

This repository contains the manual correction web tool and cell tracking algorithms for analyzing fungal cell growth from time-lapse microscopy data.

## Installation via ChatGPT

If you are a new user trying to set this up on your machine, you can copy the prompt below and paste it into ChatGPT (or another LLM) to get step-by-step interactive guidance on installing the software and configuring your local environment.

---
**Copy and paste the text below into ChatGPT:**

```text
Act as an IT installation assistant. My coworker shared a Python project with me via GitHub, and I need to set it up on my computer. I have varying levels of experience, so please guide me through these steps one by one, waiting for my confirmation after each step before proceeding. 

Here are the requirements and facts about the tool:
1. It is a Python web application running on Flask.
2. The core dependencies are: `flask`, `pandas`, `numpy`, `scikit-image`, `Pillow`, and `scipy`.
3. The codebase contains a critical web tool called `manual_correction_tool.py`.
4. The tool requires access to our raw microscopy data located on our Network Attached Storage (NAS). 
5. Inside `manual_correction_tool.py`, there is a variable called `BASE_MOVIE_ROOT`. This variable must be manually edited to match the exact absolute path where my computer mounts the NAS.

Please walk me through the following process:
Step 1: Installing Python (if I don't have it) and creating a virtual environment (e.g., using conda or venv).
Step 2: Installing the required pip dependencies.
Step 3: Finding my local mount path to the NAS (Network Attached Storage) and showing me how to edit `BASE_MOVIE_ROOT` in `manual_correction_tool.py` to point to it.
Step 4: Running the `manual_correction_tool.py` script and opening the local web server at http://127.0.0.1:5001.

Ask me if I'm on Mac or Windows first, and then let's start with Step 1!
```
---

## Manual Installation Steps

If you prefer to install it manually without an LLM assistant, follow these steps:

1. **Clone the repository**: Download this code to your local machine.
2. **Install Python & Dependencies**: Make sure you have Python 3.9+ installed. Run:
   `pip install flask pandas numpy scikit-image pillow scipy`
3. **Configure the Data Path**: Open `manual_correction_tool.py` and find `BASE_MOVIE_ROOT = Path(...)`. Change the path string to point exactly to where your computer mounts the shared NAS drive.
4. **Run**: Open a terminal in this directory and execute:
   `python manual_correction_tool.py`
5. **View**: Open `http://127.0.0.1:5001` in your browser.
