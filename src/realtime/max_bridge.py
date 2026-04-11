from pythonosc import udp_client
import time

client = udp_client.SimpleUDPClient("127.0.0.1", 8000)

def send_parameters(theta):

    brightness = float(theta[0])
    bandwidth = float(theta[1])
    texture = float(theta[2])

    client.send_message("/brightness", brightness)
    client.send_message("/bandwidth", bandwidth)
    client.send_message("/texture", texture)