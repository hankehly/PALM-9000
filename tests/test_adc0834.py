"""Tests for the ADC0834 bit-banged SPI-ish protocol.

The chip clocks its 8-bit result out twice: once MSB-first, then LSB-first.
``read`` compares the two and returns 0 when they disagree, which is the
device's built-in integrity check.
"""

import pytest

from palm_9000.adc0834 import ADC0834

CS, CLK, DIO = 17, 18, 27


def double_clocked(value: int) -> list[int]:
    """Bits as the chip emits them: MSB-first, then the same value LSB-first."""
    msb_first = [(value >> (7 - i)) & 1 for i in range(8)]
    lsb_first = [(value >> i) & 1 for i in range(8)]
    return msb_first + lsb_first


@pytest.fixture
def adc(fake_gpio):
    return ADC0834(cs=CS, clk=CLK, dio=DIO, frequency=400_000)


class TestSetup:
    def test_configures_cs_and_clk_as_outputs(self, adc, fake_gpio):
        adc.setup()
        assert (CS, "OUT") in fake_gpio.setups
        assert (CLK, "OUT") in fake_gpio.setups

    def test_returns_self_for_chaining(self, adc):
        assert adc.setup() is adc

    def test_does_not_configure_dio_yet(self, adc, fake_gpio):
        """DIO direction is flipped during read(), not during setup()."""
        adc.setup()
        assert all(pin != DIO for pin, _ in fake_gpio.setups)


class TestRead:
    @pytest.mark.parametrize("value", [0, 1, 42, 127, 128, 200, 255])
    def test_returns_the_clocked_value(self, adc, fake_gpio, value):
        fake_gpio.input_values = double_clocked(value)
        assert adc.read() == value

    def test_returns_zero_when_the_two_readings_disagree(self, adc, fake_gpio):
        # MSB-first says 0xFF, LSB-first says 0x00 -> integrity check fails.
        fake_gpio.input_values = [1] * 8 + [0] * 8
        assert adc.read() == 0

    def test_brackets_the_exchange_with_chip_select(self, adc, fake_gpio):
        fake_gpio.input_values = double_clocked(0)
        adc.read()

        cs_writes = [(pin, val) for pin, val in fake_gpio.outputs if pin == CS]
        assert cs_writes[0] == (CS, 0)  # pulled LOW to begin
        assert cs_writes[-1] == (CS, 1)  # released HIGH to end

    def test_toggles_dio_direction_around_the_read(self, adc, fake_gpio):
        fake_gpio.input_values = double_clocked(0)
        adc.read()

        dio_modes = [mode for pin, mode in fake_gpio.setups if pin == DIO]
        # OUT to send the channel selection, IN to read, OUT again at the end.
        assert dio_modes == ["OUT", "IN", "OUT"]

    def test_reads_sixteen_bits_total(self, adc, fake_gpio):
        fake_gpio.input_values = double_clocked(0)
        adc.read()
        assert sum(1 for c in fake_gpio.calls if c[0] == "input") == 16

    def test_sends_start_and_sgl_dif_bits_high(self, adc, fake_gpio):
        fake_gpio.input_values = double_clocked(0)
        adc.read(channel=0)

        dio_writes = [val for pin, val in fake_gpio.outputs if pin == DIO]
        # Start bit, then SGL/DIF, both high for single-ended mode.
        assert dio_writes[0] == 1
        assert dio_writes[1] == 1

    @pytest.mark.parametrize(
        "channel,odd_sign,select1",
        [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)],
    )
    def test_encodes_the_channel_selection(
        self, adc, fake_gpio, channel, odd_sign, select1
    ):
        fake_gpio.input_values = double_clocked(0)
        adc.read(channel=channel)

        dio_writes = [val for pin, val in fake_gpio.outputs if pin == DIO]
        # [start, sgl/dif, odd/sign, select1]
        assert dio_writes[2] == odd_sign
        assert dio_writes[3] == select1

    def test_clock_returns_low_before_switching_dio_to_input(self, adc, fake_gpio):
        """The MUX needs half a clock cycle to settle before sampling."""
        fake_gpio.input_values = double_clocked(0)
        adc.read()

        calls = fake_gpio.calls
        dio_in_index = next(i for i, c in enumerate(calls) if c == ("setup", DIO, "IN"))
        prior_clk = [
            c for c in calls[:dio_in_index] if c[0] == "output" and c[1] == CLK
        ]
        assert prior_clk[-1] == ("output", CLK, 0)


class TestClockTiming:
    def test_tick_sleeps_for_half_a_clock_period(self, fake_gpio, monkeypatch):
        slept = []
        monkeypatch.setattr("palm_9000.adc0834.time.sleep", slept.append)

        ADC0834(cs=CS, clk=CLK, dio=DIO, frequency=50_000)._tick()

        assert slept == [pytest.approx(1 / 50_000 / 2)]

    def test_higher_frequency_sleeps_less(self, fake_gpio, monkeypatch):
        slept = []
        monkeypatch.setattr("palm_9000.adc0834.time.sleep", slept.append)

        ADC0834(cs=CS, clk=CLK, dio=DIO, frequency=400_000)._tick()

        assert slept == [pytest.approx(1 / 400_000 / 2)]

    def test_default_frequency_is_within_the_supported_range(self):
        """Datasheet allows 10-400 kHz."""
        adc = ADC0834(cs=CS, clk=CLK, dio=DIO)
        assert 10_000 <= adc.frequency <= 400_000

    def test_clock_helpers_drive_the_pin(self, adc, fake_gpio):
        adc._set_clock_high()
        adc._set_clock_low()
        clk_writes = [(pin, val) for pin, val in fake_gpio.outputs if pin == CLK]
        assert clk_writes == [(CLK, 1), (CLK, 0)]


def test_constructor_stores_pin_assignments(fake_gpio):
    adc = ADC0834(cs=1, clk=2, dio=3, frequency=12_345)
    assert (adc.cs, adc.clk, adc.dio, adc.frequency) == (1, 2, 3, 12_345)
