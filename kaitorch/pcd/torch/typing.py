from typing import Union

from .pointcloudxyz import PointCloudXYZ

from .pointcloudxyzi import PointCloudXYZI
from .pointcloudxyzr import PointCloudXYZR
from .pointcloudxyztime import PointCloudXYZTime
from .pointcloudxyzid import PointCloudXYZID

from .pointcloudxyzir import PointCloudXYZIR
from .pointcloudxyzitime import PointCloudXYZITime
from .pointcloudxyziid import PointCloudXYZIID
from .pointcloudxyzrtime import PointCloudXYZRTime
from .pointcloudxyzrid import PointCloudXYZRID
from .pointcloudxyztimeid import PointCloudXYZTimeID

from .pointcloudxyzirtime import PointCloudXYZIRTime
from .pointcloudxyzirid import PointCloudXYZIRID
from .pointcloudxyzitimeid import PointCloudXYZITimeID
from .pointcloudxyzrtimeid import PointCloudXYZRTimeID

from .pointcloudxyzirtimeid import PointCloudXYZIRTimeID


PointClouds = Union[
    PointCloudXYZ,

    PointCloudXYZI, PointCloudXYZR, PointCloudXYZTime, PointCloudXYZID,

    PointCloudXYZIR, PointCloudXYZITime, PointCloudXYZIID,
    PointCloudXYZRTime, PointCloudXYZRID, PointCloudXYZTimeID,

    PointCloudXYZIRTime, PointCloudXYZIRID, PointCloudXYZITimeID,
    PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

PointCloudIs = Union[
    PointCloudXYZI,

    PointCloudXYZIR, PointCloudXYZITime, PointCloudXYZIID,

    PointCloudXYZIRTime, PointCloudXYZIRID, PointCloudXYZITimeID,

    PointCloudXYZIRTimeID
]

PointCloudRs = Union[
    PointCloudXYZR,

    PointCloudXYZIR, PointCloudXYZRTime, PointCloudXYZRID,

    PointCloudXYZIRTime, PointCloudXYZIRID, PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

PointCloudTimes = Union[
    PointCloudXYZTime,

    PointCloudXYZITime, PointCloudXYZRTime, PointCloudXYZTimeID,

    PointCloudXYZIRTime, PointCloudXYZITimeID, PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

PointCloudIDs = Union[
    PointCloudXYZID,

    PointCloudXYZIID, PointCloudXYZRID, PointCloudXYZTimeID,

    PointCloudXYZIRID, PointCloudXYZITimeID, PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

# I +
PointCloudIRs = Union[
    PointCloudXYZIR,

    PointCloudXYZIRTime, PointCloudXYZIRID,

    PointCloudXYZIRTimeID
]

PointCloudITimes = Union[
    PointCloudXYZITime,

    PointCloudXYZIRTime, PointCloudXYZITimeID,

    PointCloudXYZIRTimeID
]

PointCloudIIDs = Union[
    PointCloudXYZIID,

    PointCloudXYZIRID,

    PointCloudXYZIRTimeID
]

# R +
PointCloudRTimes = Union[
    PointCloudXYZRTime,

    PointCloudXYZIRTime, PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

PointCloudRIDs = Union[
    PointCloudXYZRID,

    PointCloudXYZIRID, PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

# Time +
PointCloudTimeIDs = Union[
    PointCloudXYZTimeID,

    PointCloudXYZITimeID, PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

# IR +
PointCloudIRTimes = Union[
    PointCloudXYZIRTime,

    PointCloudXYZIRTimeID
]

PointCloudIRIDs = Union[
    PointCloudXYZIRID,

    PointCloudXYZIRTimeID
]

# ITime +
PointCloudITimeIDs = Union[
    PointCloudXYZITimeID,

    PointCloudXYZIRTimeID
]

# RTime+
PointCloudRTimeIDs = Union[
    PointCloudXYZRTimeID,

    PointCloudXYZIRTimeID
]

# IRTime+
PointCloudIRTimeIDs = PointCloudXYZIRTimeID
