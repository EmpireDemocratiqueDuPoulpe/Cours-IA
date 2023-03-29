# Requests and net stuff
import requests
from requests.exceptions import HTTPError

# Misc.
import typing

# Local files
from config import constants


def post_payload(target_uri: str, payload: dict) -> typing.Any:
    """ Sends a POST request to the target URI. """
    try:
        resp = requests.post(target_uri, json=payload)
        resp.raise_for_status()

        constants.LOGGER.info(msg=f"Response status code: {resp.status_code}")
        constants.LOGGER.info(msg=f"Response: {resp.json()}")
        return resp
    except HTTPError as exc:
        constants.LOGGER.error(msg=f"Error: {exc}")
        raise exc


if __name__ == "__main__":
    example_payload = {
        "Source IP": "10.42.0.211",
        "Source Port": 50004,
        "Destination IP": "172.217.6.202",
        "Destination Port": 443.0,
        "Protocol": 6.0,
        "Timestamp": "13/06/2017 11:52:39",
        "Flow Duration": 37027,
        "Total Fwd Packets": 1,
        "Total Backward Packets": 1,
        "Total Length of Fwd Packets": 0.0,
        "Total Length of Bwd Packets": 0.0,
        "Fwd Packet Length Max": 0.0,
        "Fwd Packet Length Min": 0.0,
        "Fwd Packet Length Mean": 0.0,
        "Fwd Packet Length Std": 0.0,
        "Bwd Packet Length Max": 0.0,
        "Bwd Packet Length Min": 0.0,
        "Bwd Packet Length Mean": 0.0,
        "Bwd Packet Length Std": 0.0,
        "Flow Bytes/s": 0.0,
        "Flow Packets/s": 54.01463796688903,
        "Flow IAT Mean": 37027.0,
        "Flow IAT Std": 0.0,
        "Flow IAT Max": 37027.0,
        "Flow IAT Min": 37027.0,
        "Fwd IAT Total": 0.0,
        "Fwd IAT Mean": 0.0,
        "Fwd IAT Std": 0.0,
        "Fwd IAT Max": 0.0,
        "Fwd IAT Min": 0.0,
        "Bwd IAT Total": 0.0,
        "Bwd IAT Mean": 0.0,
        "Bwd IAT Std": 0.0,
        "Bwd IAT Max": 0.0,
        "Bwd IAT Min": 0.0,
        "Fwd PSH Flags": 0.0,
        "Bwd PSH Flags": 0.0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": 32,
        "Bwd Header Length": 32,
        "Fwd Packets/s": 27.00731898344452,
        "Bwd Packets/s": 27.00731898344452,
        "Min Packet Length": 0.0,
        "Max Packet Length": 0.0,
        "Packet Length Mean": 0.0,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "FIN Flag Count": 0.0,
        "SYN Flag Count": 0.0,
        "RST Flag Count": 0.0,
        "PSH Flag Count": 0.0,
        "ACK Flag Count": 1.0,
        "URG Flag Count": 1.0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": 1.0,
        "Average Packet Size": 0.0,
        "Avg Fwd Segment Size": 0.0,
        "Avg Bwd Segment Size": 0.0,
        "Fwd Header Length.1": 32.0,
        "Fwd Avg Bytes/Bulk": 0.0,
        "Fwd Avg Packets/Bulk": 0.0,
        "Fwd Avg Bulk Rate": 0.0,
        "Bwd Avg Bytes/Bulk": 0.0,
        "Bwd Avg Packets/Bulk": 0.0,
        "Bwd Avg Bulk Rate": 0.0,
        "Subflow Fwd Packets": 1.0,
        "Subflow Fwd Bytes": 0.0,
        "Subflow Bwd Packets": 1.0,
        "Subflow Bwd Bytes": 0.0,
        "Init_Win_bytes_forward": 2994.0,
        "Init_Win_bytes_backward": 362.0,
        "act_data_pkt_fwd": 0.0,
        "min_seg_size_forward": 32.0,
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0.0,
        "Idle Min": 0.0
    }

    url = "http://prediction-server:8001/predict"
    # url = "http://localhost:8000/predict"
    response = post_payload(target_uri=url, payload=example_payload)
