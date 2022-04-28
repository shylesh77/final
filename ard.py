import serial
import time
def heart_control():
    try:
        ser=serial.Serial('com9',9600)
        time.sleep(2)
        count=0
        sum1=0
        while count<5:
                ser_bytes=ser.readline()
                decoded_bytes=ser_bytes.decode("utf-8")
                print(int(decoded_bytes))
                sum1=sum1+int(decoded_bytes)
                count=count+1
    except serial.SerialException:
        return "There's no arduino board"
    return sum1/5

#heart_control()
