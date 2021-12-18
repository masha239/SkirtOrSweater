import numpy as np
import pandas as pd
import math
from PIL import Image
from collections import defaultdict


def resize_max_n(picture, n):
    img = Image.fromarray(picture)
    h, v = picture.shape[0], picture.shape[1]
    dmax = max(v, h)
    new_v = (v * n) // dmax
    new_h = (h * n) // dmax
    return np.asarray(img.resize((new_h, new_v)))


def make_square(picture):
    h = picture.shape[0]
    w = picture.shape[1]

    if h < w:
        h1 = (w - h) // 2
        h2 = (w - h) - h1
        top = np.full((h1, w), picture.mean())
        bottom = np.full((h2, w), picture.mean())
        picture = np.vstack((top, picture, bottom))

    if h > w:
        w1 = (h - w) // 2
        w2 = (h - w) - w1
        left = np.full((h, w1), picture.mean())
        right = np.full((h, w2), picture.mean())
        picture = np.hstack((left, picture, right))

    return picture


def normalize(picture):
    return (picture - np.mean(picture)) / np.std(picture)


def smooth(picture, eps=2):
    new_picture = np.zeros(picture.shape)
    for i in range(picture.shape[0]):
        for j in range(picture.shape[1]):
            new_picture[i, j] = np.mean(picture[max(0, i - eps): i + eps, max(0, j - eps): j + eps])
    return new_picture


def make_center_darker(picture):
    h, v = picture.shape[0], picture.shape[1]
    a1 = (np.mean(picture[: h // 3, : v // 3]) + np.mean(picture[: h // 3, 2 * v // 3:]) +
          np.mean(picture[2 * h // 3:, : v // 3]) + np.mean(picture[2 * h // 3:, 2 * v // 3:])) / 4
    a2 = np.mean(picture[h // 3: 2 * h // 3, v // 3: 2 * v // 3])

    if a1 > a2:
        picture = -picture
    return picture


def cut_picture(picture, step=10, alpha=0.15, sp=20):
    hs = [np.std(picture[i * step: (i + 1) * step, :]) for i in range(picture.shape[0] // step)]
    r = max(hs) - min(hs)
    q = min(hs) + alpha * r
    big_hs = [i * step for i in range(len(hs)) if hs[i] > q]
    ih_min, ih_max = max(min(big_hs) - sp, 0), min(max(big_hs) + sp, picture.shape[0])

    vs = [np.std(picture[:, i * step: (i + 1) * step]) for i in range(picture.shape[1] // step)]
    r = max(vs) - min(vs)
    q = min(vs) + alpha * r
    big_vs = [i * step for i in range(len(vs)) if vs[i] > q]
    iv_min, iv_max = max(min(big_vs) - sp, 0), min(max(big_vs) + sp, picture.shape[1])
    return picture[ih_min: ih_max, iv_min: iv_max]


def make_symmetrical(picture, step=10):
    results = []
    for i in range(picture.shape[1] // 3, 2 * picture.shape[1] // 3, step):
        s = min(i, picture.shape[1] - i)
        pic1 = picture[:, max(0, i - s):i]
        pic2 = picture[:, i: min(i + s, picture.shape[1])]
        assert pic1.shape == pic2.shape
        n1 = np.sum((pic1 - pic1.mean()) * (pic1 - pic1.mean()))
        n2 = np.sum((pic2 - pic2.mean()) * (pic2 - pic2.mean()))
        n3 = np.sum((pic1 - pic1.mean()) * (np.flip(pic2, 1) - pic2.mean()))
        res = n3 / math.sqrt(n1 * n2)
        results.append(abs(res))

    ans = picture.shape[1] // 3 + np.argmax(results) * step
    half_len = min(ans, picture.shape[1] - ans)
    picture = picture[:, ans - half_len: ans + half_len]
    return picture


def prepare_picture1(path, n=400, eps=2):
    # читаем
    picture = Image.open(path).convert('L')
    picture = np.asarray(picture)

    # обрезаем края
    picture = cut_picture(picture)

    # центрируем
    picture = make_symmetrical(picture)

    # уменьшаем еще
    picture = resize_max_n(picture, n)

    # делаем квадрат
    picture = make_square(picture)

    # сглаживаем
    picture = smooth(picture, eps=eps)

    # инвертируем при необходимости
    picture = make_center_darker(picture)

    # нормализуем
    picture = normalize(picture)

    return picture


def calculate_angles(picture, n_steps):
    eps = 0.00001
    res = [0] * n_steps
    part = math.pi / n_steps
    for i in range(1, picture.shape[0] - 1):
        for j in range(1, picture.shape[1] - 1):
            dx = picture[i + 1, j] - picture[i - 1, j]
            dy = picture[i, j + 1] - picture[i, j - 1]
            r = math.sqrt(dx ** 2 + dy ** 2)
            if r < eps:
                continue

            if dx == 0:
                phi = math.pi / 2
            else:
                phi = math.atan(dy / dx)
            a = 0
            while phi > -math.pi / 2 + a * part:
                a += 1
            pb = phi + math.pi / 2 - (a - 1) * part
            pa = part - pb
            b = a - 1
            if b < 0:
                b += n_steps
            if a == n_steps:
                a -= n_steps

            if phi != 0:
                res[b] += pb * r / part
                res[a] += pa * r / part
            else:
                res[0] += r

    return res


def make_features(pictures, hist_step=10, angle_step=80):
    df = pd.DataFrame()
    n = pictures[0].shape[0]

    # градиенты
    angles = []
    names = []
    for picture in pictures:
        ans = []
        d = defaultdict(int)
        for i in range(0, picture.shape[0], angle_step):
            for j in range(0, picture.shape[1], angle_step):
                arr = calculate_angles(picture[i: i + angle_step, j: j + angle_step], 9)
                ans += arr
                d[tuple(arr)] += 1
        # print(d)
        angles.append(ans)

    for i in range(0, picture.shape[0], angle_step):
        for j in range(0, picture.shape[1], angle_step):
            names += [f'grad_{i}_{j}_{k}' for k in range(9)]
    df = pd.DataFrame(angles, columns=names)

    # просто гистрограмма распределения цвета по вертикали и горизонтали
    for i in range(0, n - hist_step + 1, hist_step):
        res1 = []
        res2 = []
        for picture in pictures:
            res1.append(np.sum(picture[:, i: hist_step + i]))
            res2.append(np.sum(picture[i: hist_step + i, :]))

        df[f'h_{i}'] = res1
        df[f'w_{i}'] = res2

    good_features = ['grad_0_100_6', 'grad_0_300_0', 'grad_0_300_5', 'grad_200_0_8',
                     'grad_200_200_1', 'h_100', 'w_120', 'h_200', 'w_200', 'h_220', 'h_280']

    return df[good_features]
