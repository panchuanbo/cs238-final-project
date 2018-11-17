## CS 238 - Tetris AI

The goal of this project is to build a Tetris AI using methods taught in Stanford's CS 238 Class (Fall 2018 w/ Prof. Mykel Kochenderfer).

It uses the Nintaco Emulator to run Tetris, and the Nintaco API to interface with the emulator and get key parts of the program (e.g., interface, score, state, controls).

## Setup

Run `pip install -r requirements.txt` or `pip install -r requirements-gpu.txt` depending on your computer's configuration. Note that as of this README, TensorFlow 1.12.0 only supports NVIDIA GPUs. Check the TensorFlow website to see details about additional requirements when using GPU Acceleration.

## Quick Start

* Make sure you have Nintaco downloaded. There is no need to download the API because it is bundled in this project.
* Load the Tetris ROM file.
* Once the game has finished loading, go to Tools and select `Start Program Server...`
* Start the server and then run `python custom_input.py`
* You should now be able to input commands and see those commands reflected in the emulator.
