import matplotlib.pyplot as plt
import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree, Shaped, ArrayLike, Int
import equinox as eqx

class PendulumSimulation(eqx.Module):

    box_size: Float = 5.0
    ball_size: Float = 0.2
    image_size: Int = 64

    def __init__(self,
        box_size: Float = 5.0,
        ball_size: Float = 0.2,
        image_size: Int = 64
    ):
        self.box_size = box_size
        self.ball_size = ball_size
        self.image_size = image_size
        
    def ODE_system(self, t: Float, y: Float[Array, " 2"], args: PyTree) -> Float[Array, " 2"]:
        angle, angular_velocity = y
        gravity, length = args
        return [angular_velocity, -gravity / length * jnp.sin(angle)]


    def simulate_pendulum(
        self,
        initial_angle: Float,
        initial_velocity: Float,
        gravity: Float,
        length: Float,
        time: Float,
        dt: Float,
    ):
        initial_state = [initial_angle, initial_velocity]
        args = [gravity, length]
        term = dx.ODETerm(self.ODE_system)
        solver = dx.Tsit5()
        saveat = dx.SaveAt(ts=jnp.arange(0, time, dt))
        sol = dx.diffeqsolve(
            term, solver, t0=0, t1=time, y0=initial_state, args=args, dt0=dt, saveat=saveat
        )
        return sol

    def render_pendulum(
        self,
        solution: Float[Array, " 2"],
        length: Float,
    ) -> Float[Array, " n_res n_res"]:
        image = jnp.zeros((self.image_size, self.image_size)).reshape(-1)
        grid_x, grid_y = jnp.meshgrid(jnp.arange(self.image_size) - self.image_size//2, jnp.arange(self.image_size) - self.image_size//2)
        grid_x = grid_x / self.image_size * self.box_size
        grid_y = grid_y / self.image_size * self.box_size

        coordinates = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        angle, angular_velocity = solution[0], solution[1]
        position = jnp.array([length * jnp.sin(angle), length * jnp.cos(angle)])

        distance = jnp.linalg.norm(coordinates - position, axis=-1)
        pendulum_index = jnp.argwhere(distance < self.ball_size)[:, 0]
        image = image.at[pendulum_index].set(1)
        return image.reshape(self.image_size, self.image_size)
        

    def generate_dataset(
        self,
        num_samples: Int,
        time: Float,
        dt: Float,
        gravity: Float,
        length: Float,
    ) -> Float[Array, " n_samples n_res n_res"]:
        dataset = jnp.zeros((num_samples, self.image_size, self.image_size))
        for i in range(num_samples):
            initial_angle = jax.random.uniform(jax.random.PRNGKey(i), minval=-jnp.pi, maxval=jnp.pi)
            initial_velocity = jax.random.uniform(jax.random.PRNGKey(i), minval=-10.0, maxval=10.0)
            solution = self.simulate_pendulum(initial_angle, initial_velocity, gravity, length, time, dt)
            solution = solution.ys[0]
            dataset = dataset.at[i].set(self.render_pendulum(solution, length))
        return dataset

pendulum = PendulumSimulation(image_size=128)
solution = pendulum.simulate_pendulum(0.1, 0.0, 9.8, 2.0, 10.0, 0.01)
solution = solution.ys[0]
snapshot = pendulum.render_pendulum(solution, 2.0)
