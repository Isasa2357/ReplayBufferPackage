
import multiprocessing as mp
from multiprocessing import Lock
from contextlib import contextmanager

__all__ = [
    "RWLock"
]

class RWLock:
    '''
    Reader-Writer Lock
    - 複数Readerは同時に通す
    - Writerは排他的
    - Writerが書き込み中はReaderは読めない
    '''
    def __init__(self) -> None:
        self._cond = mp.Condition(Lock())
        self._readers: int = 0
        self._writerActive: bool = False
        self._writersWaiting: int = 0

    def acquireRead(self) -> None:
        with self._cond:
            while self._writerActive or self._writersWaiting > 0:
                self._cond.wait()
            self._readers += 1

    def releaseRead(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquireWrite(self) -> None:
        with self._cond:
            self._writersWaiting += 1
            while self._writerActive or self._readers > 0:
                self._cond.wait()
            self._writersWaiting -= 1
            self._writerActive = True

    def releaseWrite(self) -> None:
        with self._cond:
            self._writerActive = False
            self._cond.notify_all()

    @contextmanager
    def readLock(self):
        self.acquireRead()
        try:
            yield
        finally:
            self.releaseRead()


    @contextmanager
    def writeLock(self):
        self.acquireWrite()
        try:
            yield
        finally:
            self.releaseWrite()