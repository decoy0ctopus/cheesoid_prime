# Cheesoid Prime

## How to get GUI to work via SSH

### On client machine


$xhost + 

to disable access control

$export DISPLAY="CLIENT_IP_ADDRESS:10.0"

CLIENT_IP_ADDRESS is your client machine local ip address i.e export DISPLAY="127.0.0.1:10.0"

### On Raspberry Pi

$sudo raspi-config


-> Interface Options -> SSH Enable -> Yes
-> Interface Options -> VNC Enable -> Yes

ssh pi@PI_IP_ADDRESS -X

export DISPLAY="127.0.0.1:10.0"