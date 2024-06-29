from typing import Generic, NewType, TypeVar, Union

import torch


TorchBool = Union[torch.BoolTensor, torch.cuda.BoolTensor]

TorchInt8 = Union[torch.CharTensor, torch.cuda.CharTensor]
TorchInt16 = Union[torch.ShortTensor, torch.cuda.ShortTensor]
TorchInt32 = Union[torch.IntTensor, torch.cuda.IntTensor]
TorchInt64 = Union[torch.LongTensor, torch.cuda.LongTensor]
TorchInt = Union[TorchInt8, TorchInt16, TorchInt32, TorchInt64]

TorchUInt8 = Union[torch.ByteTensor, torch.cuda.ByteTensor]

TorchFloat16 = Union[torch.HalfTensor, torch.cuda.HalfTensor]
TorchBFloat16 = Union[torch.BFloat16Tensor, torch.cuda.BFloat16Tensor]
TorchFloat32 = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TorchFloat64 = Union[torch.DoubleTensor, torch.cuda.DoubleTensor]
TorchFloat = Union[TorchFloat16, TorchBFloat16, TorchFloat32, TorchFloat64]

TorchReal = Union[TorchInt, TorchUInt8, TorchFloat]

TorchDType = NewType('TorchDType', torch.dtype)
TorchDevice = NewType('TorchDevice', torch.device)

real = Union[int, float]

Bool = Union[bool, TorchBool]
Int = Union[int, TorchInt]
UInt = TorchUInt8
Float = Union[float, TorchFloat]
Real = Union[real, TorchReal]

TorchT = TypeVar('TorchT', bound=Union[TorchBool, TorchReal])
T = TypeVar('T', bound=Union[Bool, Real, bool, real])


class TorchTensor(Generic[TorchT], torch.Tensor):
    ...


class TorchTensorLike(Generic[T]):
    ...
