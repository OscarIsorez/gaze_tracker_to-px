void Start()
{
    var receiver = gameObject.AddComponent<GazeReceiver>();
    receiver.OnGazeReceived += (screenX, screenY, timestamp) =>
    {
        // Use screenX, screenY (0..1920,0..1080) as needed, e.g. move a UI cursor:
        Vector3 uiPos = new Vector3(screenX, Screen.height - screenY, 0);
        cursorRectTransform.position = uiPos;
    };
}
