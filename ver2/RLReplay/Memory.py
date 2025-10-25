
from typing import Tuple, List

import numpy as np

import torch
from torch import Tensor

from RLReplay.Param import MultipleReplayMemoyParam
from RLReplay.SamplingSumTree_py.SamplingSumTree import SamplingSumTree

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
    
    def realSize(self) -> int:
        return self._realSize

    def capcity(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return self._capacity
    

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

    def to(self, device: torch.device):
        '''
        デバイスの変更
        '''
        self._status = self._status.to(device)
        self._actions = self._actions.to(device)
        self._rewards = self._rewards.to(device)
        self._nextStatus = self._nextStatus.to(device)
        self._dones = self._dones.to(device)
    
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
