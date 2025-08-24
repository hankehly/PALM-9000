import time

import RPi.GPIO as GPIO


class ADC0834:
    """
    A class representing the ADC0834 Analog-to-Digital Converter.

    Args:
        cs (int): The chip select GPIO pin number.
        clk (int): The clock GPIO pin number.
        dio (int): The data input/output GPIO pin number.
        frequency (int): The frequency of the clock signal in Hz.
            The acceptable range is 10-400 kHz (10,000 - 400,000 Hz)
    """

    def __init__(self, cs: int, clk: int, dio: int, frequency: int = 50_000) -> None:
        self.cs = cs
        self.clk = clk
        self.dio = dio
        self.frequency = frequency

    def setup(self) -> "ADC0834":
        GPIO.setup(self.cs, GPIO.OUT)
        GPIO.setup(self.clk, GPIO.OUT)
        return self

    def read(self, channel: int = 0) -> int:
        """
        Read the value from the specified channel.
        Returns an int between 0 and 255.
        """
        # Set CS pin to low to enable the ADC
        GPIO.output(self.cs, GPIO.LOW)

        # Set DIO pin to output to setup the ADC to read from the specified channel
        GPIO.setup(self.dio, GPIO.OUT)

        # Start bit
        self._set_clock_low()
        GPIO.output(self.dio, 1)
        self._set_clock_high()

        # SGL/DIF
        self._set_clock_low()
        GPIO.output(self.dio, 1)
        self._set_clock_high()

        # ODD/SIGN
        self._set_clock_low()
        GPIO.output(self.dio, channel % 2)
        self._set_clock_high()

        # SELECT1
        self._set_clock_low()
        GPIO.output(self.dio, int(channel > 1))
        self._set_clock_high()

        # Allow the MUX to settle for 1/2 clock cycle
        self._set_clock_low()

        # Switch DIO pin to input to read data
        GPIO.setup(self.dio, GPIO.IN)

        # Read data from MSB to LSB
        val1 = 0
        for i in range(0, 8):
            self._set_clock_high()
            self._set_clock_low()
            val1 = val1 << 1
            val1 = val1 | GPIO.input(self.dio)

        # Read data from LSB to MSB
        val2 = 0
        for i in range(0, 8):
            bit = GPIO.input(self.dio) << i
            val2 = val2 | bit
            self._set_clock_high()
            self._set_clock_low()

        # Set CS pin to high to clear all internal registers
        GPIO.output(self.cs, GPIO.HIGH)

        # Done reading, set DIO pin back to output
        GPIO.setup(self.dio, GPIO.OUT)

        # Compare the two values to ensure they match
        if val1 == val2:
            return val1
        else:
            return 0

    def _set_clock_high(self):
        GPIO.output(self.clk, GPIO.HIGH)
        self._tick()

    def _set_clock_low(self):
        GPIO.output(self.clk, GPIO.LOW)
        self._tick()

    def _tick(self):
        period = 1 / self.frequency
        period_half = period / 2
        time.sleep(period_half)
