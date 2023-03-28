# Misc.
import random
from locust import HttpUser, task, between


class User(HttpUser):
    wait_time = between(1, 5)

    @task
    def make_prediction(self):
        payload = {
            "Source IP": f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "Source Port": random.randint(0, 100_000),
            "Destination IP": f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "Destination Port": random.randint(0, 100_000),
            "Protocol": random.randint(0, 10),
            "Timestamp": "13/06/2017 11:52:39",
            "Flow Duration": random.randint(0, 100_000),
            "Total Fwd Packets": random.randint(0, 1),
            "Total Backward Packets": random.randint(0, 1),
            "Total Length of Fwd Packets": random.randint(0, 1),
            "Total Length of Bwd Packets": random.randint(0, 1),
            "Fwd Packet Length Max": random.randint(0, 1),
            "Fwd Packet Length Min": random.randint(0, 1),
            "Fwd Packet Length Mean": random.randint(0, 1),
            "Fwd Packet Length Std": random.randint(0, 1),
            "Bwd Packet Length Max": random.randint(0, 1),
            "Bwd Packet Length Min": random.randint(0, 1),
            "Bwd Packet Length Mean": random.randint(0, 1),
            "Bwd Packet Length Std": random.randint(0, 1),
            "Flow Bytes/s": random.randint(0, 1),
            "Flow Packets/s": random.randint(0, 100),
            "Flow IAT Mean": random.randint(0, 100_000),
            "Flow IAT Std": random.randint(0, 1),
            "Flow IAT Max": random.randint(0, 100_000),
            "Flow IAT Min": random.randint(0, 100_000),
            "Fwd IAT Total": random.randint(0, 1),
            "Fwd IAT Mean": random.randint(0, 1),
            "Fwd IAT Std": random.randint(0, 1),
            "Fwd IAT Max": random.randint(0, 1),
            "Fwd IAT Min": random.randint(0, 1),
            "Bwd IAT Total": random.randint(0, 1),
            "Bwd IAT Mean": random.randint(0, 1),
            "Bwd IAT Std": random.randint(0, 1),
            "Bwd IAT Max": random.randint(0, 1),
            "Bwd IAT Min": random.randint(0, 1),
            "Fwd PSH Flags": random.randint(0, 1),
            "Bwd PSH Flags": random.randint(0, 1),
            "Fwd URG Flags": random.randint(0, 1),
            "Bwd URG Flags": random.randint(0, 1),
            "Fwd Header Length": random.randint(0, 100),
            "Bwd Header Length": random.randint(0, 100),
            "Fwd Packets/s": random.randint(0, 100),
            "Bwd Packets/s": random.randint(0, 100),
            "Min Packet Length": random.randint(0, 1),
            "Max Packet Length": random.randint(0, 1),
            "Packet Length Mean": random.randint(0, 1),
            "Packet Length Std": random.randint(0, 1),
            "Packet Length Variance": random.randint(0, 1),
            "FIN Flag Count": random.randint(0, 1),
            "SYN Flag Count": random.randint(0, 1),
            "RST Flag Count": random.randint(0, 1),
            "PSH Flag Count": random.randint(0, 1),
            "ACK Flag Count": random.randint(0, 1),
            "URG Flag Count": random.randint(0, 1),
            "CWE Flag Count": random.randint(0, 1),
            "ECE Flag Count": random.randint(0, 1),
            "Down/Up Ratio": random.randint(0, 1),
            "Average Packet Size": random.randint(0, 1),
            "Avg Fwd Segment Size": random.randint(0, 1),
            "Avg Bwd Segment Size": random.randint(0, 1),
            "Fwd Header Length.1": random.randint(0, 100),
            "Fwd Avg Bytes/Bulk": random.randint(0, 1),
            "Fwd Avg Packets/Bulk": random.randint(0, 1),
            "Fwd Avg Bulk Rate": random.randint(0, 1),
            "Bwd Avg Bytes/Bulk": random.randint(0, 1),
            "Bwd Avg Packets/Bulk": random.randint(0, 1),
            "Bwd Avg Bulk Rate": random.randint(0, 1),
            "Subflow Fwd Packets": random.randint(0, 1),
            "Subflow Fwd Bytes": random.randint(0, 1),
            "Subflow Bwd Packets": random.randint(0, 1),
            "Subflow Bwd Bytes": random.randint(0, 1),
            "Init_Win_bytes_forward": random.randint(0, 10_000),
            "Init_Win_bytes_backward": random.randint(0, 1_000),
            "act_data_pkt_fwd": random.randint(0, 1),
            "min_seg_size_forward": random.randint(0, 100),
            "Active Mean": random.randint(0, 1),
            "Active Std": random.randint(0, 1),
            "Active Max": random.randint(0, 1),
            "Active Min": random.randint(0, 1),
            "Idle Mean": random.randint(0, 1),
            "Idle Std": random.randint(0, 1),
            "Idle Max": random.randint(0, 1),
            "Idle Min": random.randint(0, 1)
        }

        self.client.post("/predict", json=payload)
