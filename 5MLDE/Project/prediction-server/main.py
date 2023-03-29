# OS and filesystem
import sys
sys.path.append("..")

# Request and net stuff
from fastapi import FastAPI, Request

# Misc.
from pydantic import BaseModel, Field

# Local files
from lib.modelling import load_pipeline, run_inference
from internal_config import LOGGER, MODEL_VERSION, APP_TITLE, APP_DESCRIPTION, APP_VERSION, LAST_MODEL_PATH


# App initialization
app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


class InputData(BaseModel):
    SourceIP: str = Field(alias="Source IP")
    SourcePort: int = Field(alias="Source Port")
    DestinationIP: str = Field(alias="Destination IP")
    DestinationPort: int = Field(alias="Destination Port")
    Protocol: float = Field(alias="Protocol")
    Timestamp: str = Field(alias="Timestamp")
    FlowDuration: float = Field(alias="Flow Duration")
    TotalFwdPackets: float = Field(alias="Total Fwd Packets")
    TotalBackwardPackets: float = Field(alias="Total Backward Packets")
    TotalLengthofFwdPackets: float = Field(alias="Total Length of Fwd Packets")
    TotalLengthofBwdPackets: float = Field(alias="Total Length of Bwd Packets")
    FwdPacketLengthMax: float = Field(alias="Fwd Packet Length Max")
    FwdPacketLengthMin: float = Field(alias="Fwd Packet Length Min")
    FwdPacketLengthMean: float = Field(alias="Fwd Packet Length Mean")
    FwdPacketLengthStd: float = Field(alias="Fwd Packet Length Std")
    BwdPacketLengthMax: float = Field(alias="Bwd Packet Length Max")
    BwdPacketLengthMin: float = Field(alias="Bwd Packet Length Min")
    BwdPacketLengthMean: float = Field(alias="Bwd Packet Length Mean")
    BwdPacketLengthStd: float = Field(alias="Bwd Packet Length Std")
    FlowBytesPerS: float = Field(alias="Flow Bytes/s")
    FlowPacketsPerS: float = Field(alias="Flow Packets/s")
    FlowIATMean: float = Field(alias="Flow IAT Mean")
    FlowIATStd: float = Field(alias="Flow IAT Std")
    FlowIATMax: float = Field(alias="Flow IAT Max")
    FlowIATMin: float = Field(alias="Flow IAT Min")
    FwdIATTotal: float = Field(alias="Fwd IAT Total")
    FwdIATMean: float = Field(alias="Fwd IAT Mean")
    FwdIATStd: float = Field(alias="Fwd IAT Std")
    FwdIATMax: float = Field(alias="Fwd IAT Max")
    FwdIATMin: float = Field(alias="Fwd IAT Min")
    BwdIATTotal: float = Field(alias="Bwd IAT Total")
    BwdIATMean: float = Field(alias="Bwd IAT Mean")
    BwdIATStd: float = Field(alias="Bwd IAT Std")
    BwdIATMax: float = Field(alias="Bwd IAT Max")
    BwdIATMin: float = Field(alias="Bwd IAT Min")
    FwdPSHFlags: float = Field(alias="Fwd PSH Flags")
    BwdPSHFlags: float = Field(alias="Bwd PSH Flags")
    FwdURGFlags: float = Field(alias="Fwd URG Flags")
    BwdURGFlags: float = Field(alias="Bwd URG Flags")
    FwdHeaderLength: float = Field(alias="Fwd Header Length")
    BwdHeaderLength: float = Field(alias="Bwd Header Length")
    FwdPacketsPerS: float = Field(alias="Fwd Packets/s")
    BwdPacketsPerS: float = Field(alias="Bwd Packets/s")
    MinPacketLength: float = Field(alias="Min Packet Length")
    MaxPacketLength: float = Field(alias="Max Packet Length")
    PacketLengthMean: float = Field(alias="Packet Length Mean")
    PacketLengthStd: float = Field(alias="Packet Length Std")
    PacketLengthVariance: float = Field(alias="Packet Length Variance")
    FINFlagCount: float = Field(alias="FIN Flag Count")
    SYNFlagCount: float = Field(alias="SYN Flag Count")
    RSTFlagCount: float = Field(alias="RST Flag Count")
    PSHFlagCount: float = Field(alias="PSH Flag Count")
    ACKFlagCount: float = Field(alias="ACK Flag Count")
    URGFlagCount: float = Field(alias="URG Flag Count")
    CWEFlagCount: float = Field(alias="CWE Flag Count")
    ECEFlagCount: float = Field(alias="ECE Flag Count")
    DownUpRatio: float = Field(alias="Down/Up Ratio")
    AveragePacketSize: float = Field(alias="Average Packet Size")
    AvgFwdSegmentSize: float = Field(alias="Avg Fwd Segment Size")
    AvgBwdSegmentSize: float = Field(alias="Avg Bwd Segment Size")
    FwdHeaderLengthdot1: float = Field(alias="Fwd Header Length.1")
    FwdAvgBytesPerBulk: float = Field(alias="Fwd Avg Bytes/Bulk")
    FwdAvgPacketsPerBulk: float = Field(alias="Fwd Avg Packets/Bulk")
    FwdAvgBulkRate: float = Field(alias="Fwd Avg Bulk Rate")
    BwdAvgBytesPerBulk: float = Field(alias="Bwd Avg Bytes/Bulk")
    BwdAvgPacketsPerBulk: float = Field(alias="Bwd Avg Packets/Bulk")
    BwdAvgBulkRate: float = Field(alias="Bwd Avg Bulk Rate")
    SubflowFwdPackets: float = Field(alias="Subflow Fwd Packets")
    SubflowFwdBytes: float = Field(alias="Subflow Fwd Bytes")
    SubflowBwdPackets: float = Field(alias="Subflow Bwd Packets")
    SubflowBwdBytes: float = Field(alias="Subflow Bwd Bytes")
    Init_Win_bytes_forward: float = Field(alias="Init_Win_bytes_forward")
    Init_Win_bytes_backward: float = Field(alias="Init_Win_bytes_backward")
    act_data_pkt_fwd: float = Field(alias="act_data_pkt_fwd")
    min_seg_size_forward: float = Field(alias="min_seg_size_forward")
    ActiveMean: float = Field(alias="Active Mean")
    ActiveStd: float = Field(alias="Active Std")
    ActiveMax: float = Field(alias="Active Max")
    ActiveMin: float = Field(alias="Active Min")
    IdleMean: float = Field(alias="Idle Mean")
    IdleStd: float = Field(alias="Idle Std")
    IdleMax: float = Field(alias="Idle Max")
    IdleMin: float = Field(alias="Idle Min")


class PredictionOut(BaseModel):
    infection_type: str


pipeline = load_pipeline(path=LAST_MODEL_PATH, logger=LOGGER)


async def print_request(request):
    print("Incoming request")
    print(f"    > Header         : {dict(request.headers.items())}")
    print(f"    > Query params   : {dict(request.query_params.items())}")
    try : 
        print(f"    > JSON           : {await request.json()}")
    except Exception as err:
        print(f"    > Body           : {await request.body()}")


# Route set up
@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_VERSION, "is this field useful?": "no"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
async def predict(payload: Request):  # The Input type is not working
    #await print_request(payload)
    # infection_type = run_inference(pipeline=pipeline, payload=payload.dict(), logger=LOGGER)
    infection_type = run_inference(pipeline=pipeline, payload=(await payload.json()), logger=LOGGER)
    return {"infection_type": infection_type}
