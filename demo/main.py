"""Pulseq-zero demo: optimize TSE refocusing flip angles to minimize SAR.

Builds a 2D TSE sequence (see ``write_tse.py``), simulates a target image
with a fully-180° refocusing train, then optimizes a per-echo flip-angle
vector so the reconstruction stays close to the target (RMS loss on the
magnitude image) while the SAR proxy ``sum(flips**2)`` goes down.

The sequence is rebuilt from the current flip angles in every iteration.
``check_timing()`` translates the whole sequence through PyPulseq, so it runs
once before the loop instead of in every rebuild.
"""

import os
from time import time

import matplotlib.pyplot as plt
import MRzeroCore as mr0
import numpy as np
import torch

# Force write_tse to use pulseq-zero:
import pulseqzero
import sys
sys.modules["pypulseq"] = pulseqzero
from write_tse import main as build_tse  # noqa: E402


N_ECHO = 16
N_ITER = 30
TARGET_AMPLITUDE = 0.8
INITIAL_FLIP_DEG = 160.0
PHANTOM_PATH = os.path.join(os.path.dirname(__file__), 'brain.npz')


def simulate(flips, data):
    seq = build_tse(refoc_flips=flips, check_timing=False).to_mr0()

    graph = mr0.compute_graph(seq, data)
    signal = mr0.execute_graph(graph, seq, data, print_progress=False)
    return mr0.reco_adjoint(
        signal, seq.get_kspace(), (64, 64, 1), (0.256, 0.256, 1)
    )


def main():
    data = mr0.VoxelGridPhantom.load(PHANTOM_PATH).slices([36]).build()
    build_tse(check_timing=True)  # same timing for every flip angle train

    # Target image: the full-180° refocusing train.
    with torch.no_grad():
        target_flips = torch.full((N_ECHO,), np.pi)
        target = TARGET_AMPLITUDE * simulate(target_flips, data)

    # Start the optimization from the same initial flip train.
    flips = torch.full((N_ECHO,), INITIAL_FLIP_DEG * np.pi / 180, requires_grad=True)
    optimizer = torch.optim.Adam([flips], lr=0.02)

    start = simulate(flips, data).detach()
    RMS_WEIGHT = 1 / ((start.abs() - target.abs()) ** 2).mean().sqrt()
    SAR_WEIGHT = 1 / (flips.detach() ** 2).sum()

    data_hist = []
    sar_hist = []
    loss_hist = []
    flip_hist = []

    t0 = time()
    for i in range(N_ITER):
        t_iter = time()
        optimizer.zero_grad()
        reco = simulate(flips, data)

        data_loss = ((reco.abs() - target.abs()) ** 2).mean().sqrt()
        sar_loss = (flips ** 2).sum()
        loss = RMS_WEIGHT * data_loss + SAR_WEIGHT * sar_loss

        loss.backward()
        optimizer.step()

        data_hist.append(data_loss.item())
        sar_hist.append(sar_loss.item())
        loss_hist.append(loss.item())
        flip_hist.append(flips.detach().clone().numpy())
        print(
            f'{i + 1}/{N_ITER}: data={data_loss.item():.4f}, '
            f'SAR={sar_loss.item():.2f}, total={loss.item():.4f}, '
            f'{time() - t_iter:.2f} s'
        )
    print(f'Optimization took {time() - t0:.1f} s')

    best_idx = int(np.argmin(loss_hist))
    best = simulate(torch.as_tensor(flip_hist[best_idx]), data).detach()

    plt.figure(figsize=(10, 8), dpi=120)
    plt.subplot(2, 2, 1)
    plt.title('Target (all 180°)')
    mr0.util.imshow(target.abs(), vmin=0, cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title(f'Best (iter {best_idx + 1})')
    mr0.util.imshow(best.abs(), vmin=0, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('Loss')
    plt.plot([RMS_WEIGHT * d for d in data_hist], label='RMS image loss')
    plt.plot([SAR_WEIGHT * s for s in sar_hist], label='SAR loss')
    plt.plot(loss_hist, label='total', linestyle='--')
    plt.xlabel('iteration')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title('Refocusing flip angles')
    cmap = plt.get_cmap('viridis')
    for i, flips in enumerate(flip_hist):
        plt.plot(
            [f * 180 / np.pi for f in flips],
            color=cmap(i / (N_ITER - 1)),
        )
    plt.xlabel('iteration')
    plt.ylabel('Flip [°]')
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
