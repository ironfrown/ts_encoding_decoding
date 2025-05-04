#!/bin/bash

# A sequence of steps
python qtsa_00_data_2sin_create_v1.2.py    log_3 -d 2_sins    -f Target_2_sins -st 70 -sv 30       -rs 1410
python qtsa_00_data_2sin_sw_create_v1.2.py log_3 -d 2_sins_sw -f Target_2_sins -st 70 -sv 30 -wn 5 -rs 2024
