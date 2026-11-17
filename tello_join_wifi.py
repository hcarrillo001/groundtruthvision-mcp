"""Put a Tello EDU on your home Wi-Fi (station mode). Run once.

1. Turn the Tello on and join this computer to its Wi-Fi (TELLO-XXXXXX).
2. python tello_join_wifi.py "Home WiFi name" "password"
3. The Tello restarts and joins your Wi-Fi. Rejoin your home Wi-Fi on this
   computer and find the Tello's address in your router's device list
   (it shows as a new device, often named after its Espressif Wi-Fi chip).
4. Set TELLO_HOST to that address. Reserve it in the router so it never changes.

Your network must be 2.4 GHz: the Tello can't join 5 GHz-only networks.
To undo, hold the Tello's power button for 5 seconds to reset it.
"""
import sys

from djitellopy import Tello

if len(sys.argv) != 3:
    sys.exit(__doc__)
t = Tello()
t.connect()
print("Battery:", t.get_battery(), "%")
t.connect_to_wifi(sys.argv[1], sys.argv[2])
print("Sent. The Tello will restart and join", repr(sys.argv[1]))
