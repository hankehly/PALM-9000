import time

import RPi.GPIO as GPIO


class ADC0834ReadError(RuntimeError):
    """The chip's two readings of the same conversion disagreed."""


class ADC0834:
    """
    A class representing the ADC0834 Analog-to-Digital Converter.

    Args:
        cs (int): The chip select GPIO pin number.
        clk (int): The clock GPIO pin number.
        dio (int): The data input/output GPIO pin number.
        frequency (int): The frequency of the clock signal in Hz.
            The acceptable range is 10-400 kHz (10,000 - 400,000 Hz)

    Pin numbers are BCM. `setup()` selects BCM mode if the caller has not
    already chosen one, and refuses to run under BOARD mode, where these
    numbers would silently address the wrong pins.
    """

    def __init__(self, cs: int, clk: int, dio: int, frequency: int = 50_000) -> None:
        self.cs = cs
        self.clk = clk
        self.dio = dio
        self.frequency = frequency

    @property
    def _half_period(self) -> float:
        return 1 / self.frequency / 2

    def setup(self) -> "ADC0834":
        """Configure the pins. Selects BCM numbering if none is set yet.

        RPi.GPIO requires a pin-numbering mode before any setup() call, and
        raises an unhelpful error if none has been chosen. Doing it here means
        callers do not have to remember `GPIO.setmode(GPIO.BCM)` first.
        """
        mode = GPIO.getmode()
        if mode is None:
            GPIO.setmode(GPIO.BCM)
        elif mode != GPIO.BCM:
            raise RuntimeError(
                "ADC0834 pin numbers are BCM, but GPIO numbering is already "
                "set to BOARD. Re-run with GPIO.setmode(GPIO.BCM), or convert "
                "the pin numbers."
            )

        GPIO.setup(self.cs, GPIO.OUT)
        GPIO.setup(self.clk, GPIO.OUT)
        return self

    def read(self, channel: int = 0) -> int:
        """
        Read the value from the specified channel.

        Returns an int between 0 and 255.

        The chip clocks the same conversion out twice, MSB-first then
        LSB-first. If the two disagree the reading is corrupt -- usually
        loose wiring or too high a clock frequency -- and this raises
        ADC0834ReadError. It previously returned 0 in that case, which was
        indistinguishable from a genuine zero reading; for a moisture sensor
        that meant a wiring fault looked like bone-dry soil.
        """
        # Set CS pin to low to enable the ADC
        GPIO.output(self.cs, GPIO.LOW)

        # Set DIO pin to output to setup the ADC to read from the specified channel
        GPIO.setup(self.dio, GPIO.OUT)

        self._write_bit(1)  # Start bit
        self._write_bit(1)  # SGL/DIF: single-ended
        self._write_bit(channel % 2)  # ODD/SIGN
        self._write_bit(int(channel > 1))  # SELECT1

        # Allow the MUX to settle for 1/2 clock cycle
        self._set_clock_low()

        # Switch DIO pin to input to read data
        GPIO.setup(self.dio, GPIO.IN)

        # Read data from MSB to LSB
        msb_first = 0
        for _ in range(0, 8):
            self._set_clock_high()
            self._set_clock_low()
            msb_first = msb_first << 1
            msb_first = msb_first | GPIO.input(self.dio)

        # Read data from LSB to MSB
        lsb_first = 0
        for i in range(0, 8):
            bit = GPIO.input(self.dio) << i
            lsb_first = lsb_first | bit
            self._set_clock_high()
            self._set_clock_low()

        # Set CS pin to high to clear all internal registers
        GPIO.output(self.cs, GPIO.HIGH)

        # Done reading, set DIO pin back to output
        GPIO.setup(self.dio, GPIO.OUT)

        # Compare the two values to ensure they match
        if msb_first != lsb_first:
            raise ADC0834ReadError(
                f"channel {channel} readings disagree: "
                f"{msb_first} (MSB-first) vs {lsb_first} (LSB-first)"
            )
        return msb_first

    def _write_bit(self, value: int) -> None:
        """Clock one bit out to the chip on the falling-then-rising edge."""
        self._set_clock_low()
        GPIO.output(self.dio, value)
        self._set_clock_high()

    def _set_clock_high(self):
        GPIO.output(self.clk, GPIO.HIGH)
        self._tick()

    def _set_clock_low(self):
        GPIO.output(self.clk, GPIO.LOW)
        self._tick()

    def _tick(self):
        time.sleep(self._half_period)
