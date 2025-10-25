
from typing import Tuple, List, Dict, Union, Sequence, cast

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor

from collections import defaultdict

import multiprocessing as mp
from multiprocessing.context import SpawnProcess as mpSpwanProcess
from multiprocessing import Lock, Queue, Condition, Value
from multiprocessing import Process as mpProcess
from multiprocessing.synchronize import Lock as mpLock
from multiprocessing.synchronize import Event as mpEvent
from multiprocessing.synchronize import Semaphore as mpSemaphore
from multiprocessing.sharedctypes import Synchronized as mpSynchronized
from multiprocessing.sharedctypes import SynchronizedArray as mpSynchronizedArray
import time

from RLReplay.Param import MultipleReplayMemoyParam
from RLReplay.SamplingSumTree_py.SamplingSumTree import SamplingSumTree
from RLReplay.RWLock import *

IndicesType = Union[Sequence[int], NDArray, torch.Tensor]

class BaseReplayBuffer:

    def __init__(self, 
                 capacity: int, 
                 stateShape: Tuple, actionShape: Tuple,  
                 stateType: torch.dtype, actionType: torch.dtype, rewardType: torch.dtype, doneType: torch.dtype, 
                 device: torch.device):
        '''
        リプレイバッファの基底クラス

        Args:
            capacity: リプレイバッファのサイズ
            stateShape: stateのShape
            actionShape: actionのShape
            stateType: stateの型
            actionType: actionの型
            rewardType: rewardの型
            doneType: doneの型
            device: リプレイバッファのデバイス
        '''

        self._capacity = capacity
        self._device = device

        self._status = torch.empty(tuple([capacity]) + stateShape, dtype=stateType,device=self._device)
        self._actions = torch.empty(tuple([capacity]) + actionShape, dtype=actionType, device=self._device)
        self._rewards = torch.empty(tuple([capacity, 1]), dtype=torch.float16, device=self._device)
        self._nextStatus = torch.empty(tuple([capacity]) + stateShape, dtype=stateType,device=self._device)
        self._dones = torch.empty(tuple([capacity, 1]), dtype=torch.int8, device=self._device)

        self._realSize = 0
        self._writeIndex = 0

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, nextState: torch.Tensor, done: torch.Tensor) -> None:
        raise NotImplementedError
    
    def _stepWriteIndex(self):
        '''
        writeIndexを進める
        '''
        self._writeIndex = (self._writeIndex + 1) % self._capacity

    def _stepRealSize(self):
        '''
        realSizeを進める
        '''
        self._realSize = min(self._realSize + 1, self._capacity)
    
    @property
    def realSize(self) -> int:
        return self._realSize
    
    @property
    def writeIndex(self) -> int:
        return self._writeIndex

    def capcity(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return self._capacity
    
    def isFull(self):
        return self._realSize == self._capacity
    
    def to(self, device: torch.device):
        '''
        デバイスの変更
        '''
        self._status = self._status.to(device)
        self._actions = self._actions.to(device)
        self._rewards = self._rewards.to(device)
        self._nextStatus = self._nextStatus.to(device)
        self._dones = self._dones.to(device)
    

class ReplayBuffer(BaseReplayBuffer):

    def __init__(self, 
                 capacity: int, 
                 stateShape: Tuple, actionShape: Tuple, 
                 stateType: torch.dtype, actionType: torch.dtype, rewardType: torch.dtype, doneType: torch.dtype, 
                 device: torch.device):
        '''
        一様サンプリングリプレイバッファ

        Args:
            capacity: リプレイバッファのサイズ
            stateShape: stateのShape
            actionShape: actionのShape
            stateType: stateの型
            actionType: actionの型
            rewardType: rewardの型
            doneType: doneの型
            device: リプレイバッファのデバイス
        '''
        super().__init__(capacity, 
                         stateShape, actionShape, 
                         stateType, actionType, rewardType, doneType, 
                         device)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, nextState: torch.Tensor, done: torch.Tensor) -> None:
        '''
        バッファに経験を追加する
        '''
        
        self._status[self._writeIndex] = state
        self._actions[self._writeIndex] = action
        self._rewards[self._writeIndex] = reward
        self._nextStatus[self._writeIndex] = nextState
        self._dones[self._writeIndex] = done

        self._stepWriteIndex()
        self._stepRealSize()

    def getBatch(self, batchSize: int) -> list[torch.Tensor]:
        
        sampleIndices = np.random.randint(0, self._realSize, batchSize)

        sampleStatus = self._status[sampleIndices]
        sampleActions = self._actions[sampleIndices]
        sampleRewards = self._rewards[sampleIndices]
        sampleNextStatus = self._nextStatus[sampleIndices]
        sampleDones = self._dones[sampleIndices]

        return [sampleStatus, sampleActions, sampleRewards, sampleNextStatus, sampleDones]
    
class PERBuffer(BaseReplayBuffer):

    def __init__(self, 
                 capacity: int, 
                 stateShape: Tuple, actionShape: Tuple, 
                 stateType: torch.dtype, actionType: torch.dtype, rewardType: torch.dtype, doneType: torch.dtype, 
                 alpha: MultipleReplayMemoyParam, beta: MultipleReplayMemoyParam, 
                 device: torch.device):
        '''
        優先度付きリプレイバッファ
        '''
        super().__init__(capacity, 
                         stateShape, actionShape, 
                         stateType, actionType, rewardType, doneType, 
                         device)
        
        self._alpha = alpha
        self._beta = beta

        self._prioritiesTree = SamplingSumTree(capacity)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, nextState: torch.Tensor, done: torch.Tensor) -> None:
        '''
        バッファに経験を追加する
        '''
        
        self._status[self._writeIndex] = state
        self._actions[self._writeIndex] = action
        self._rewards[self._writeIndex] = reward
        self._nextStatus[self._writeIndex] = nextState
        self._dones[self._writeIndex] = done

        # print(f"real size: {self._realSize}")
        # print(f"initial priority {float(self._initialPriority())}")
        self._prioritiesTree.add(float(self._initialPriority()))

        self._stepWriteIndex()
        self._stepRealSize()
    
    def _initialPriority(self) -> np.float64:
        if self._realSize == 0:
            return np.float64(1.0)
        else:
            return self._prioritiesTree.total()
    
    def getBatch(self, batchSize: int) -> List[torch.Tensor]:
        priorities, sampleIndices = self._prioritiesTree.randomWeightedSampling(batchSize)

        prioritiesTensor = torch.tensor(priorities, dtype=torch.float64, device=self._device)
        sampleIndicesTensor = torch.tensor(sampleIndices, dtype=torch.int16, device=self._device)

        sampleStatus = self._status[sampleIndices]
        sampleActions = self._actions[sampleIndices]
        sampleRewards = self._rewards[sampleIndices]
        sampleNextStatus = self._nextStatus[sampleIndices]
        sampleDones = self._dones[sampleIndices]

        # print(f"prorities: {priorities}")
        # print(f"index: {sampleIndices}")
        weights = self._calcWeights(prioritiesTensor)

        return [sampleStatus, sampleActions, sampleRewards, sampleNextStatus, sampleDones, weights, sampleIndicesTensor]

    def updatePriorities(self, tdDiffs: Tensor, indices: Tensor):
        # print(f"td diff: {tdDiffs}")
        newPriorities = self._calcPriorities(tdDiffs).detach().cpu().numpy()
        # print(f"new priorities: {newPriorities}")
        indicesNP = indices.detach().cpu().numpy()
        self._prioritiesTree.multiWrite(newPriorities, indicesNP)

    def _calcWeights(self, priorities: Tensor) -> torch.Tensor:
        priorityTotal = torch.tensor(self._prioritiesTree.total(), dtype=torch.float64, device=self._device)
        selectProbs = priorities / priorityTotal
        weights = (self._prioritiesTree.realSize() * selectProbs)**self._beta.value()
        maxWeight = torch.max(weights)
        # print(f"weights: {weights}")
        # print(f"max weight: {maxWeight}")
        return weights / maxWeight

    def _calcPriorities(self, tdDiffs: Tensor) -> torch.Tensor:
        priorities = ((tdDiffs + 1e-6)**self._alpha.value())
        return priorities
    
    def stepParam(self):
        self._alpha.step()
        self._beta.step()

def add_queue_pop_worker(add_q: mp.Queue, chunk_queues: List[mp.Queue],
                         chunk_size: int, div_num: int, stop_ev: mpEvent, qSizeLogger: List[mpSynchronized]):
    def get_chunk_id(index: int) -> int:
        # return min(index // chunk_size, div_num - 1)
        return index % div_num

    while not stop_ev.is_set():
        try:
            idx, obs = add_q.get(timeout=0.01)  # obs = [state, action, reward, nextState, done]
        except Exception:
            continue
        chunk_id = get_chunk_id(idx)
        chunk_queues[chunk_id].put([idx, *obs])
        with qSizeLogger[chunk_id].get_lock():
            qSizeLogger[chunk_id].value += 1

def chunk_add_queue_pop_worker(
    chunk_id: int, chunk_q: mp.Queue,
    status_sh, actions_sh, rewards_sh, nextstatus_sh, dones_sh, 
    stop_ev: mpEvent, lock: mpLock,
    realSize: mpSynchronized, maxRealSize: int, writerLimiter: mpSemaphore, 
    qSizeLogger: mpSynchronized
):
    from queue import Empty

    def step_real_size() -> None:
        # 共有カウンタをロックしてその場で更新（戻り値は使わない）
        with realSize.get_lock():
            if realSize.value < maxRealSize:
                realSize.value += 1

    local_batch = []
    SENTINEL = None

    while True:
        if stop_ev.is_set() and not local_batch and chunk_q.empty():
            break

        while True:
            try:
                item = chunk_q.get(timeout=0.01)
                with qSizeLogger.get_lock():
                    qSizeLogger.value -= 1
                if item is SENTINEL:
                    # 残りを吐いてから抜ける
                    pass
                else:
                    local_batch.append(item)
            except Empty:
                break

        if local_batch or stop_ev.is_set():
            writerLimiter.acquire()
            try:
                with lock:
                    for idx, state, action, reward, nextState, done in local_batch:
                        status_sh[idx]     = state
                        actions_sh[idx]    = action
                        rewards_sh[idx]    = reward
                        nextstatus_sh[idx] = nextState
                        dones_sh[idx]      = done
                        step_real_size()
                local_batch.clear()
            finally:
                writerLimiter.release()
                time.sleep(0.1)

        if stop_ev.is_set() and chunk_q.empty() and not local_batch:
            break

class MultiAccessableReplayBuffer(BaseReplayBuffer):
    def __init__(self, 
                 capacity: int, divNum: int,
                 stateShape: Tuple, actionShape: Tuple, 
                 stateType: torch.dtype, actionType: torch.dtype, rewardType: torch.dtype, doneType: torch.dtype):
        if divNum > capacity:
            raise Exception("capacityはdivNumより大きい必要があります")

        super().__init__(capacity, 
                         stateShape, actionShape, 
                         stateType, actionType, rewardType, doneType, 
                         torch.device('cpu'))

        self._realSize = mp.Value("i", 0)

        self._divNum = divNum
        self._chunkSize = self._capacity // self._divNum

        # ---- multiprocessing コンテキスト ----
        self._ctx = mp.get_context("spawn")

        # 共有リソース（まずは単純化：チャンク毎に単一Lock）
        self._chunkLocks = [self._ctx.Lock() for _ in range(self._divNum)]
        self._maxChunkAddQueueWriterN = self._divNum // 3
        self._writerLimiter = self._ctx.Semaphore(self._maxChunkAddQueueWriterN)
        self._addLock = self._ctx.Lock()

        # 停止フラグは Event で
        self._stop = self._ctx.Event()

        # プロセス間Queue
        self._addQueue = self._ctx.Queue()
        self._chunkAddQueues = [self._ctx.Queue() for _ in range(self._divNum)]
        self._chunkAddQueuesSize = [mp.Value("i", 0) for _ in range(self._divNum)]

        # ---- 子プロセスから書けるように共有メモリ化 ----
        # BaseReplayBuffer 側で確保しているテンソルを共有メモリ化
        self._status.share_memory_()
        self._actions.share_memory_()
        self._rewards.share_memory_()
        self._nextStatus.share_memory_()
        self._dones.share_memory_()

        # ワーカー起動（**selfを渡さない**）
        self._addQueuePopProcess = self._ctx.Process(
            target=add_queue_pop_worker,
            args=(self._addQueue, self._chunkAddQueues, self._chunkSize, self._divNum, self._stop, self._chunkAddQueuesSize)
        )

        # チャンクごとのポッパー（必要数だけ）
        self._chunkAddQueuesPopProcesses: List[mpSpwanProcess] = []
        for chunk_id in range(self._divNum):
            p = self._ctx.Process(
                target=chunk_add_queue_pop_worker,
                args=(chunk_id, self._chunkAddQueues[chunk_id],
                      self._status, self._actions, self._rewards, self._nextStatus, self._dones,
                      self._stop, self._chunkLocks[chunk_id], 
                      self._realSize, self._capacity, self._writerLimiter, self._chunkAddQueuesSize[chunk_id])
            )
            self._chunkAddQueuesPopProcesses.append(p)

        self._addQueuePopProcess.start()
        for p in self._chunkAddQueuesPopProcesses:
            p.start()

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, nextState: torch.Tensor, done: torch.Tensor) -> None:
        with self._addLock:
            self._addQueue.put([self._writeIndex, [state, action, reward, nextState, done]])
            self._stepWriteIndex()

    def MultiAdd(self, status: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, nextStatus: torch.Tensor, dones: torch.Tensor) -> None:
        for i in range(len(status)):
            self.add(status[i], actions[i], rewards[i], nextStatus[i], dones[i])
    
    def _makeGetBatchTasksDict(self, indices: IndicesType) -> Dict[int, List[List[int]]]:
        tasksDict = defaultdict(lambda: [list(), list()])

        for endex, index in enumerate(indices):
            index = cast(int, index)
            chunkId = self._getChunkId(index)
            tasksDict[chunkId][0].append(index)
            tasksDict[chunkId][1].append(endex)
        return tasksDict

    def getBatch(self, batchSize: int) -> List[torch.Tensor]:
        batchIndices = torch.randint(0, self.realSize - 1, (batchSize, ))
        tasksDict = self._makeGetBatchTasksDict(batchIndices)

        ret = [
            torch.empty((batchSize, *self._status.shape[1:]),     dtype=self._status.dtype,     device=self._status.device), 
            torch.empty((batchSize, *self._actions.shape[1:]),    dtype=self._actions.dtype,    device=self._actions.device), 
            torch.empty((batchSize, *self._rewards.shape[1:]),    dtype=self._rewards.dtype,    device=self._rewards.device), 
            torch.empty((batchSize, *self._nextStatus.shape[1:]), dtype=self._nextStatus.dtype, device=self._nextStatus.device), 
            torch.empty((batchSize, *self._dones.shape[1:]),      dtype=self._dones.dtype,      device=self._dones.device)
        ]
        for chunkId, (tensorIdx, retIdx) in tasksDict.items():
            with self._chunkLocks[chunkId]:
                ret[0][retIdx] = self._status[tensorIdx]
                ret[1][retIdx] = self._actions[tensorIdx]
                ret[2][retIdx] = self._rewards[tensorIdx]
                ret[3][retIdx] = self._nextStatus[tensorIdx]
                ret[4][retIdx] = self._dones[tensorIdx]
        return ret

    def terminate(self):
        self._stop.set()
        self._addQueuePopProcess.join()
        for p in self._chunkAddQueuesPopProcesses:
            p.join()

    @property
    def divNum(self) -> int:
        return self._divNum
    
    @property
    def chunkSize(self) -> int:
        return self._chunkSize

    @property
    def terminateFlag(self) -> bool:
        return self._stop.is_set()
    
    @property
    def realSize(self) -> int:
        with self._realSize.get_lock():
            return self._realSize.value

    def _stepRealSize(self):
        with self._realSize.get_lock():
            super()._stepRealSize()
    
    def _getChunkId(self, index: int):
        '''
        チャンクIDの決定
        '''
        # return min(index // self._chunkSize, self._divNum - 1)
        return index % self._divNum
    
    def showAddQueueInfo(self) -> str:
        return f'add queue size: {self._addQueue.qsize()}'

    def showChunkAddQueueInfo(self) -> str:
        sizes = []
        for size in self._chunkAddQueuesSize:
            with size.get_lock():
                sizes.append(size.value)
        
        return f'chunk add queue size: {",".join(str(size) for size in sizes)}'
