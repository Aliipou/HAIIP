# HAIIP EconomicDecision — output of the Economic Decision Engine
# Published by: EconomicNode
# Topic: /haiip/decision/{machine_id}
#
# Action values (string enum):
#   "repair_now"  P(failure) >= 0.75  → triggers STOP command
#   "schedule"    P(failure) >= 0.50  → triggers SLOW_DOWN command
#   "monitor"     anomaly_score>=0.20 → triggers MONITOR command
#   "ignore"      below noise floor   → NOMINAL operation

std_msgs/Header header

string  machine_id
string  action                 # "repair_now" | "schedule" | "monitor" | "ignore"
float64 net_benefit_eur        # E[cost_wait] - E[cost_action]  (positive = act now)
float64 expected_cost_wait     # EUR — cost if we do nothing
float64 expected_cost_action   # EUR — cost if we maintain now
float64 confidence             # ML model confidence that fed this decision
bool    requires_human_review  # EU AI Act Art. 14 escalation flag
string  explanation            # human-readable rationale (Art. 13 transparency)
