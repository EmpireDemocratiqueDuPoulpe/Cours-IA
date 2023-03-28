# Request and net stuff
from fastapi import FastAPI

# Misc.
from pydantic import BaseModel

# Local files
from ..lib.modelling import load_pipeline, run_inference
from .internal_config import LOGGER, MODEL_VERSION, APP_TITLE, APP_DESCRIPTION, APP_VERSION, LAST_MODEL_PATH


# App initialization
app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


class InputData(BaseModel):
    SourceIP: str
    SourcePort: int
    DestinationIP: str
    DestinationPort: int
    Protocol: float
    Timestamp: str
    FlowDuration: float
    TotalFwdPackets: float
    TotalBackwardPackets: float
    TotalLengthofFwdPackets: float
    TotalLengthofBwdPackets: float
    FwdPacketLengthMax: float
    FwdPacketLengthMin: float
    FwdPacketLengthMean: float
    FwdPacketLengthStd: float
    BwdPacketLengthMax: float
    BwdPacketLengthMin: float
    BwdPacketLengthMean: float
    BwdPacketLengthStd: float
    FlowBytesPerS: float
    FlowPacketsPerS: float
    FlowIATMean: float
    FlowIATStd: float
    FlowIATMax: float
    FlowIATMin: float
    FwdIATTotal: float
    FwdIATMean: float
    FwdIATStd: float
    FwdIATMax: float
    FwdIATMin: float
    BwdIATTotal: float
    BwdIATMean: float
    BwdIATStd: float
    BwdIATMax: float
    BwdIATMin: float
    FwdPSHFlags: float
    BwdPSHFlags: float
    FwdURGFlags: float
    BwdURGFlags: float
    FwdHeaderLength: float
    BwdHeaderLength: float
    FwdPacketsPerS: float
    BwdPacketsPerS: float
    MinPacketLength: float
    MaxPacketLength: float
    PacketLengthMean: float
    PacketLengthStd: float
    PacketLengthVariance: float
    FINFlagCount: float
    SYNFlagCount: float
    RSTFlagCount: float
    PSHFlagCount: float
    ACKFlagCount: float
    URGFlagCount: float
    CWEFlagCount: float
    ECEFlagCount: float
    DownUpRatio: float
    AveragePacketSize: float
    AvgFwdSegmentSize: float
    AvgBwdSegmentSize: float
    FwdHeaderLengthdot1: float
    FwdAvgBytesPerBulk: float
    FwdAvgPacketsPerBulk: float
    FwdAvgBulkRate: float
    BwdAvgBytesPerBulk: float
    BwdAvgPacketsPerBulk: float
    BwdAvgBulkRate: float
    SubflowFwdPackets: float
    SubflowFwdBytes: float
    SubflowBwdPackets: float
    SubflowBwdBytes: float
    Init_Win_bytes_forward: float
    Init_Win_bytes_backward: float
    act_data_pkt_fwd: float
    min_seg_size_forward: float
    ActiveMean: float
    ActiveStd: float
    ActiveMax: float
    ActiveMin: float
    IdleMean: float
    IdleStd: float
    IdleMax: float
    IdleMin: float


class PredictionOut(BaseModel):
    infection_type: str


pipeline = load_pipeline(path=LAST_MODEL_PATH, logger=LOGGER)


# Route set up
@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_VERSION, "is this field useful?": "no"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData):
    infection_type = run_inference(pipeline=pipeline, payload=payload.dict(), logger=LOGGER)
    return {"infection_type": infection_type}
