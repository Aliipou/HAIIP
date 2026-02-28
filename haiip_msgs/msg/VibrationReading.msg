# HAIIP VibrationReading — 3-axis vibration from an industrial sensor
# Published by: VibrationPublisher @ 50 Hz
# Topic: /haiip/vibration/{machine_id}

std_msgs/Header header        # timestamp + frame_id = machine_id

string machine_id             # e.g. "pump-01"
float64 vib_x                 # m/s²  drive axis
float64 vib_y                 # m/s²  cross-axis
float64 vib_z                 # m/s²  axial / bearing
float64 vib_rms               # m/s²  sqrt(x²+y²+z²)
float64 rotational_speed      # RPM
float64 torque                # Nm
bool    fault_injected        # true when running in fault-mode simulation
