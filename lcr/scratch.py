import numpy as np

if __name__ == "__main__":
    y = [0, 1, 2, 3, 4, 5]
    yt = [0, 2, 2, 3, 3, 5]

    print(sum(np.array(y)==np.array(yt)) / len(np.array(yt)))