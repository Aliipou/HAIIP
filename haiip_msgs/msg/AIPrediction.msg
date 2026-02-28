# HAIIP AIPrediction — output of the anomaly detection inference node
# Published by: InferenceNode
# Topic: /haiip/ai/{machine_id}

std_msgs/Header header

string machine_id
string label                  # "ANOMALY" | "NORMAL" | "UNKNOWN"
float64 confidence            # [0, 1]
float64 anomaly_score         # [0, 1]  IsolationForest score
float64 failure_probability   # [0, 1]  GBT classifier P(failure)
bool    requires_human_review # true when confidence < threshold (EU AI Act Art.14)
bool    offline               # true when inference ran on synthetic data
