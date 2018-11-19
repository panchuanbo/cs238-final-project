import numpy as np

# Constants
class MemAddr:
    # Piece counts
    T_Count = 0x03f0
    J_Count = 0x03f2
    Z_Count = 0x03f4
    O_Count = 0x03f6
    S_Count = 0x03f8
    L_Count = 0x03fa
    I_Count = 0x03fc

    # Location
    X_Loc = 0x0040
    Y_Loc = 0x0041

    # Board
    Board_Start = 0x0400
    Board_End   = 0x04c7

    # Score
    Score_Low = 0x0073
    Score_Mid = 0x0074
    Score_Hig = 0x0075

class Const:
    Board_Width  = 10
    Board_Height = 20

# Piece Orientations
kOrientations = {
    0x00: np.array([
        [0, 1, 0],
        [1, 2, 1]
    ]), # T - up
    0x01: np.array([
        [1, 0],
        [2, 1],
        [1, 0]
    ]), # T - right
    0x02: np.array([
        [1, 2, 1],
        [0, 1, 0]
    ]), # T - down
    0x03: np.array([
        [0, 1],
        [1, 2],
        [0, 1]
    ]), # T - left
    0x04: np.array([
        [0, 1],
        [0, 2],
        [1, 1]
    ]), # flipped L - left
    0x05: np.array([
        [1, 0, 0],
        [1, 2, 1]
    ]), # flipped L - up
    0x06: np.array([
        [1, 1],
        [2, 0],
        [1, 0]
    ]), # flipped L - right
    0x07: np.array([
        [1, 2, 1],
        [0, 0, 1]
    ]), # flipped L - down
    0x08: np.array([
        [1, 2, 0],
        [0, 1, 1]
    ]), # flippped S - side
    0x09: np.array([
        [0, 1],
        [2, 1],
        [1, 0]
    ]), # flipped S - up
    0x0a: np.array([
        [1, 2],
        [1, 1]
    ]), # O
    0x0b: np.array([
        [0, 2, 1],
        [1, 1, 0]
    ]), # S - side
    0x0c: np.array([
        [1, 0],
        [2, 1],
        [0, 1]
    ]), # S - up
    0x0d: np.array([
        [1, 0],
        [2, 0],
        [1, 1]
    ]), # L - right
    0x0e: np.array([
        [1, 2, 1],
        [1, 0, 0]
    ]), # L - down
    0x0f: np.array([
        [1, 1],
        [0, 2],
        [0, 1]
    ]), # L - left
    0x10: np.array([
        [0, 0, 1],
        [1, 2, 1]
    ]), # L - up
    0x11: np.array([
        [1],
        [1],
        [2],
        [1]
    ]), # I - up
    0x12: np.array([
        [1, 1, 2, 1]
    ]) # I - side
}

# https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
