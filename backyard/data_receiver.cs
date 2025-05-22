using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class GazeReceiver : MonoBehaviour
{
    UdpClient client;
    Thread receiveThread;
    volatile bool running;
    public Action<double, float, float> OnGazeReceived;  // delegate or event to handle data

    void Start()
    {
        client = new UdpClient(5005);
        running = true;
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveLoop()
    {
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                byte[] data = client.Receive(ref anyIP);
                if (data.Length >= 24)
                {
                    // Parse binary data (little-endian double, float, float, float, float)
                    double ts = BitConverter.ToDouble(data, 0);
                    float gazeX = BitConverter.ToSingle(data, 8);
                    float gazeY = BitConverter.ToSingle(data, 12);
                    float screenX = BitConverter.ToSingle(data, 16);
                    float screenY = BitConverter.ToSingle(data, 20);

                    // Invoke any handler on main thread (store to volatile fields or use UnityMainThreadDispatcher)
                    OnGazeReceived?.Invoke(screenX, screenY, ts);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Gaze UDP receive error: " + ex);
            }
        }
    }

    void OnDestroy()
    {
        running = false;
        client.Close();
    }
}
