import jax
import jax.numpy as jnp


def square_pulse(x, start, length, amplitude):
    xprime = x - start
    return amplitude * jnp.where((xprime >= 0) & (xprime <= length), 1, 0)


def triangle_pulse(x, start, length, amplitude):
    xprime = x - start
    return amplitude * jnp.where(
        (xprime >= 0) & (xprime <= length / 2),
        2 * xprime / length,
        jnp.where(
            (xprime > length / 2) & (xprime <= length),
            2 - 2 * xprime / length,
            0,
        ),
    )


class ForcingMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def drift(self, t, y, args):
        out = super().drift(t, y, args)
        forcing = self.forcing_function(t, y, args)
        return out + forcing


class SineForcingMixin(ForcingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_amplitude = 10
        self.base_frequency = 10

    def forcing_function(self, t, y, args):
        return args["amplitude"] * jnp.sin(args["frequency"] * t + args["phase"])

    def get_initial_conditions(self, key):
        key, args_key = jax.random.split(key, 2)
        amp_key, freq_key, phase_key = jax.random.split(args_key, 3)
        shape = (self.dataset_size, 1)
        amplitude = self.base_amplitude + jax.random.uniform(amp_key, shape=shape)
        frequency = self.base_frequency + jax.random.uniform(freq_key, shape=shape)
        phase = 2 * jnp.pi * jax.random.uniform(phase_key, shape=shape)
        self.args = {
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase,
        }
        return super().get_initial_conditions(key)


class SquarePulseForcingMixin(ForcingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_amplitude = 20
        self.base_length = 2
        self.base_start = 2
        self.can_be_negative = True

    def forcing_function(self, t, y, args):
        return square_pulse(t, args["start"], args["length"], args["amplitude"])

    def get_initial_conditions(self, key):

        key, args_key = jax.random.split(key, 2)
        amp_key, length_key, start_key = jax.random.split(args_key, 3)

        shape = (self.dataset_size, 1)
        amp_noise = jax.random.uniform(amp_key, shape=shape)
        if self.can_be_negative:
            amplitude = self.base_amplitude * (2 * amp_noise - 1)
        else:
            amplitude = self.base_amplitude * amp_noise
        length = self.base_length * jax.random.uniform(length_key, shape=shape)
        start = self.base_start * jax.random.uniform(start_key, shape=shape)
        self.args = {
            "amplitude": amplitude,
            "length": length,
            "start": start,
        }

        return super().get_initial_conditions(key)


class TriangleDropForcingMixin(ForcingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_amplitude = 10
        self.base_length = 1
        self.base_start = 2

    def forcing_function(self, t, y, args):
        return args["amplitude"] - triangle_pulse(
            t, args["start"], args["length"], args["amplitude"]
        )

    def get_initial_conditions(self, key):

        key, args_key = jax.random.split(key, 2)
        amp_key, length_key, start_key = jax.random.split(args_key, 3)

        shape = (self.dataset_size, 1)
        amp_noise = jax.random.uniform(amp_key, shape=shape) - 0.5
        amplitude = self.base_amplitude + amp_noise
        length_noise = jax.random.uniform(length_key, shape=shape) - 0.5
        length = self.base_length + length_noise
        start = self.base_start * jax.random.uniform(start_key, shape=shape)
        self.args = {
            "amplitude": amplitude,
            "length": length,
            "start": start,
        }

        return super().get_initial_conditions(key)
