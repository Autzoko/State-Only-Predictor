"""LIBERO task envs in subprocesses (one per task), driven in lockstep from the main process.

Keeps a single policy copy on the GPU (large VLAs cannot be replicated per worker) and batches inference
across tasks.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(conn, sopt_data: str, suite: str, task_lang: str):
    from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks

    setup_env_vars(sopt_data)
    env = LiberoTaskEnv(suite, suite_tasks(suite).index(task_lang))
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            conn.send(env.reset(arg, seed=arg))
        elif cmd == "step":
            conn.send(env.step(arg))
        elif cmd == "close":
            env.close()
            conn.close()
            return


class TaskVecEnv:
    def __init__(self, sopt_data: str, tasks: list[str], suite: str = "libero_90"):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for t in tasks:
            a, b = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(b, str(sopt_data), suite, t), daemon=True)
            proc.start()
            self.conns.append(a)
            self.procs.append(proc)

    def reset(self, init_index: int) -> list[dict]:
        for c in self.conns:
            c.send(("reset", init_index))
        return [c.recv() for c in self.conns]

    def step(self, actions: dict[int, np.ndarray]) -> dict[int, tuple[dict, bool]]:
        """actions: {env index: 7-d sim action} for the active envs only."""
        for i, a in actions.items():
            self.conns[i].send(("step", a))
        return {i: self.conns[i].recv() for i in actions}

    def close(self) -> None:
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=30)
