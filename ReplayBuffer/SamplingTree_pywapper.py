from ReplayBuffer.SamplingTree import SamplingTree

import numpy as np
from numpy import ndarray
from nptyping import NDArray, Float
from typing import Any, List, Tuple
from functools import singledispatchmethod

class SamplingTree_pywapper:
    def __init__(self, capacity):
        # SamplingTree(cpp)
        self._tree_cpp = SamplingTree(capacity)    
    
    @singledispatchmethod
    def add(self, value) -> Any:
        '''
        SamplingTreeに値を追加する．
        TODO: cppからaddは加えた位置を返すように変更する
        '''
        raise NotImplementedError(f"Unsupported type: {type(value)}")

    @add.register
    def _(self, value: float) -> int:
        idx = self._tree_cpp.add(value)
        return idx

    @add.register
    def _(self, value: NDArray[Any, Float]) -> List[int]:
        value_lst = value.tolist()
        indics = self._tree_cpp.add_multiple(value_lst)

        return indics

    def get_sample(self, sample_size: int) -> Tuple[ndarray, ndarray]:
        '''
        treeからsample_size個のサンプリングを行う．
        重複は許可
        '''
        samples, indics = self._tree_cpp.get_smaples(sample_size)

        return np.array(samples), np.array(indics)
    
    def update(self, new_value: ndarray, indics: ndarray) -> None:
        '''
        葉の値を更新する．
        '''
        self._tree_cpp.update(new_value.tolist(), indics.tolist())
    
    @property
    def total(self) -> float:
        '''
        葉の合計
        '''
        return self._tree_cpp.total()
    
    @property
    def max_leaf(self) -> float:
        '''
        葉の中で最大値であるものの値
        '''
        return self._tree_cpp.max_leaf()