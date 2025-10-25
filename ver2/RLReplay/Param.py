
import torch

class BaseReplayMemoryParam:
    def __init__(self, start: float, minValue: float, maxValue: float, device: torch.device):
        '''
        パラメータの基底クラス
        '''
        self._device = device

        self._value = torch.tensor(start, dtype=torch.float16, device=self._device)
        self._min = torch.tensor(minValue, dtype=torch.float16, device=self._device)
        self._max = torch.tensor(maxValue, dtype=torch.float16, device=self._device)
    
    def _adjustValue(self):
        self._value = torch.max(self._value, self._min)
        self._value = torch.min(self._value, self._max)
    
    def value(self) -> torch.Tensor:
        return self._value
    
    def step(self):
        raise NotImplementedError
    
class MultipleReplayMemoyParam(BaseReplayMemoryParam):

    def __init__(self, start: float, minValue: float, maxValue: float, multiple: float, device: torch.device):
        '''
        n倍ステップパラメータ
        '''
        super().__init__(start, minValue, maxValue, device)

        self._multiple = multiple
    
    def step(self):
        self._value *= self._multiple
        self._adjustValue()