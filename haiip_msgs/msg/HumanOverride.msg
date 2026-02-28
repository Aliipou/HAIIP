# HAIIP HumanOverride — operator override command
# Published by: HumanOverridePublisher or operator CLI
# Topic: /haiip/override/{machine_id}
#
# Command values: STOP | SLOW_DOWN | MONITOR | NOMINAL | RELEASE
# RELEASE cancels any active override and returns to AI control.

std_msgs/Header header

string machine_id
string command      # "STOP" | "SLOW_DOWN" | "MONITOR" | "NOMINAL" | "RELEASE"
string reason       # operator note
string operator_id  # optional — who issued the override (audit trail)
