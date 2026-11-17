"""owlsightMCP: fly a small drone from any MCP client, safely.

Layers, from the bottom up:
  drone      a Drone driver: SimDrone (no hardware) or TelloDrone (Tello EDU)
  safety     SafetyGate: every command is checked here before it reaches a driver
  perception Detector (spot candidates every frame) and Verifier (hover and look
             closer with a vision model before believing a detection)
  mission    Mission: ties them together and keeps the flight log
  server     the MCP tools Claude calls
"""
