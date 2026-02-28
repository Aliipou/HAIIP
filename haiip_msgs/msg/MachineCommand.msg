# HAIIP MachineCommand — final actuation command sent to machine controller
# Published by: ActionNode
# Topic: /haiip/command/{machine_id}
#
# Command values:
#   "STOP"       immediate safe shutdown (repair_now)
#   "SLOW_DOWN"  reduce load (schedule)
#   "MONITOR"    increase sensor polling (monitor)
#   "NOMINAL"    normal operation (ignore)
#
# Source values:
#   "ai"     command derived from AI + economic decision
#   "human"  command set by operator override (Art. 14)

std_msgs/Header header

string machine_id
string command       # "STOP" | "SLOW_DOWN" | "MONITOR" | "NOMINAL"
string source        # "ai" | "human"
string reason        # one-line explanation
bool   override      # true when human override is active
float64 override_ttl_remaining  # seconds until override expires (0 if source=ai)
