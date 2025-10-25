import gymnasium as gym

import torch

import numpy as np

import time
import multiprocessing as mp
import threading

from RLReplay.Memory import MultiAccessableReplayBuffer

def addToExperience():
    import gymnasium as gym
    env = gym.make("CartPole-v1")

    buf = MultiAccessableReplayBuffer(20000, 50, 
                                      (4, ), (1, ), 
                                      torch.float, torch.int, torch.int, torch.int)
    
    from tqdm import tqdm
    episodes = 50000
    try:
        for episode in tqdm(range(episodes)):
            done = False
            state, _ = env.reset()

            while not done:
                action = np.random.choice([0, 1])

                nextState, rewrad, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buf.add(
                    torch.tensor(state), 
                    torch.tensor(action), 
                    torch.tensor(rewrad), 
                    torch.tensor(nextState), 
                    torch.tensor(done)
                )

                state = nextState
            # tqdm.write(f"buf realSize: {buf.realSize}")
    finally:
        buf.terminate()

# test_cartpole_5000eps_multiadd_mp.py
import time
import threading
import multiprocessing as mp

import gymnasium as gym
import torch

# ---- あなたの実装をインポート ----
# from your_module import MultiAccessableReplayBuffer

SENTINEL = ("__STOP__",)

def env_worker(out_q: mp.Queue, episodes: int, seed: int):
    """
    子プロセス：CartPole を episodes 回プレイし、(s, a, r, s', done) を逐次 out_q へ送る。
    """
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)

    for _ in range(episodes):
        status = []
        actions = []
        rewards = []
        nextStatus = []
        dones = []
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Tensor(CPU)
            # s   = torch.tensor(obs,        dtype=torch.float32)          # (4,)
            # a   = torch.tensor([action],   dtype=torch.int64)            # (1,)
            # r   = torch.tensor([reward],   dtype=torch.float32)          # (1,)
            # sp  = torch.tensor(next_obs,   dtype=torch.float32)          # (4,)
            # dn  = torch.tensor([done],     dtype=torch.int8)             # (1,)
            status.append(torch.tensor(obs,        dtype=torch.float32))          # (4,)
            actions.append(torch.tensor([action],   dtype=torch.int64))           # (1,)
            rewards.append(torch.tensor([reward],   dtype=torch.float32))         # (1,)
            nextStatus.append(torch.tensor(next_obs,   dtype=torch.float32))      # (4,)
            dones.append(torch.tensor([done],     dtype=torch.int8))              # (1,)

            # out_q.put((s, a, r, sp, dn))
            obs = next_obs
        
        for epState, epAction, epReward, epNextState, epDone in zip(status, actions, rewards, nextStatus, dones):
            out_q.put([epState, epAction, epReward, epNextState, epDone])

        # 次エピソードへ
        obs, _ = env.reset()

    out_q.put(SENTINEL)
    env.close()


def start_ingestor_thread(in_q: mp.Queue,
                          replay: "MultiAccessableReplayBuffer",
                          stop_ev: threading.Event,
                          num_workers: int,
                          flush_batch_size: int = 8192,
                          flush_interval_sec: float = 0.5):
    """
    親側：Queueから受け取り、一定件数/時間ごとに MultiAdd で投入。
    """
    buf_s, buf_a, buf_r, buf_sp, buf_dn = [], [], [], [], []
    recv_sentinels = 0
    last_flush = time.time()

    def flush(force: bool = False):
        nonlocal buf_s, buf_a, buf_r, buf_sp, buf_dn, last_flush
        if not buf_s:
            last_flush = time.time()
            return
        if force or len(buf_s) >= flush_batch_size or (time.time() - last_flush) >= flush_interval_sec:
            states      = torch.stack(buf_s,  dim=0)   # (N, 4)
            actions     = torch.stack(buf_a,  dim=0)   # (N, 1)
            rewards     = torch.stack(buf_r,  dim=0)   # (N, 1)
            next_states = torch.stack(buf_sp, dim=0)   # (N, 4)
            dones       = torch.stack(buf_dn, dim=0)   # (N, 1)
            replay.MultiAdd(states, actions, rewards, next_states, dones)
            buf_s.clear(); buf_a.clear(); buf_r.clear(); buf_sp.clear(); buf_dn.clear()
            last_flush = time.time()

    def run():
        nonlocal recv_sentinels
        while not stop_ev.is_set():
            try:
                item = in_q.get(timeout=0.05)
            except Exception:
                flush(force=False)
                continue

            if item == SENTINEL:
                recv_sentinels += 1
                flush(force=True)
                if recv_sentinels >= num_workers:
                    break
                continue

            s, a, r, sp, dn = item
            buf_s.append(s); buf_a.append(a); buf_r.append(r); buf_sp.append(sp); buf_dn.append(dn)
            flush(force=False)

        flush(force=True)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def main():
    mp.set_start_method("spawn", force=True)

    # ------- 設定 -------
    NUM_WORKERS = 100           # ワーカー数はお好みで
    EPISODES_PER_WORKER = 5000  # ← 要求どおり「各ワーカー 5000 エピソード」
    CAPACITY = 1_000_000        # リプレイ容量（上書き前提で大きめ）
    DIV_NUM = 20                # チャンク数
    FLUSH_BATCH_SIZE = 1000     # MultiAdd の件数トリガ
    FLUSH_INTERVAL = 0.5        # 時間トリガ（秒）

    # ------- リプレイバッファ作成（親） -------
    state_shape  = (4,)
    action_shape = (1,)
    state_dtype  = torch.float32
    action_dtype = torch.int64
    reward_dtype = torch.float32
    done_dtype   = torch.int8

    replay = MultiAccessableReplayBuffer(
        capacity=CAPACITY, divNum=DIV_NUM,
        stateShape=state_shape, actionShape=action_shape,
        stateType=state_dtype, actionType=action_dtype,
        rewardType=reward_dtype, doneType=done_dtype
    )

    # ------- イングレッサ（親の取り込みスレッド） -------
    q = mp.Queue(maxsize=50_000)  # バックプレッシャ用
    stop_ev = threading.Event()
    ingestor_th = start_ingestor_thread(
        q, replay, stop_ev,
        num_workers=NUM_WORKERS,
        flush_batch_size=FLUSH_BATCH_SIZE,
        flush_interval_sec=FLUSH_INTERVAL
    )

    # ------- ワーカー起動 -------
    procs = []
    base_seed = 4242
    for w in range(NUM_WORKERS):
        p = mp.Process(target=env_worker, args=(q, EPISODES_PER_WORKER, base_seed + w))
        p.start()
        procs.append(p)

    # ------- 進捗ログ（概算） -------
    try:
        last_log = time.time()
        beforeRealSize = 0
        while any(p.is_alive() for p in procs):
            fromBeforeCheck = time.time() - last_log
            if fromBeforeCheck > 2.0:
                print(f'queue realSize: {replay.realSize}')
                print(f'write index: {replay.writeIndex}')
                print(replay.showAddQueueInfo())
                print(replay.showChunkAddQueueInfo())
                print(f'speed: {(replay.realSize - beforeRealSize) / fromBeforeCheck} exp/s')
                print()

                beforeRealSize = replay.realSize

                # 共有 realSize があれば表示（なければスキップ）
                rs = getattr(replay, "_realSize_mp", None)
                if rs is not None:
                    print(f"[info] real size ≈ {rs.value}")
                last_log = time.time()
            time.sleep(0.1)
    finally:
        for p in procs:
            p.join()
        ingestor_th.join(timeout=120)
        stop_ev.set()

    # ------- 取り出しテスト -------
    try:
        batch = replay.getBatch(1024)
        s, a, r, sp, dn = batch
        print("Batch shapes:", s.shape, a.shape, r.shape, sp.shape, dn.shape)
    except Exception as e:
        print("getBatch failed:", e)

    # ------- クリーンアップ -------
    replay.terminate()
    try:
        q.close(); q.join_thread()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
