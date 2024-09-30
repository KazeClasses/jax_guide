import matplotlib.pyplot as plt
import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree, Shaped, ArrayLike


def ODE_system(t: Float, y: Float[Array, " 2"], args: PyTree) -> Float[Array, " 2"]:
    angle, angular_velocity = y
    gravity, length = args
    return [angular_velocity, -gravity / length * jnp.sin(angle)]


def simulate_pendulum(
    initial_angle: Float,
    initial_velocity: Float,
    gravity: Float,
    length: Float,
    time: Float,
    dt: Float,
):
    initial_state = [initial_angle, initial_velocity]
    args = [gravity, length]
    term = dx.ODETerm(ODE_system)
    solver = dx.Tsit5()
    saveat = dx.SaveAt(ts=jnp.arange(0, time, dt))
    sol = dx.diffeqsolve(
        term, solver, t0=0, t1=time, y0=initial_state, args=args, dt0=dt, saveat=saveat
    )
    return sol



# def render_pendulum(
#     solution: ArrayLike[Float, "2"],
#     length: Float,
#     image_size: Float,
#     pendulum_size: Float = 5,
# ) -> None:

solution = simulate_pendulum(0.1, 0.0, 9.8, 1.0, 10.0, 0.01)
solution = solution.ys[0]

length = 1.0
image_size = 64

image = jnp.zeros((image_size, image_size))
grid_x, grid_y = jnp.meshgrid(jnp.arange(image_size) - image_size//2, jnp.arange(image_size) - image_size//2)

coordinates = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

angle, angular_velocity = solution[0], solution[1]
position = jnp.array([length * jnp.sin(angle), -length * jnp.cos(angle)])

distance = jnp.linalg.norm(coordinates - position, axis=-1)

# distance = 

# plt.plot(angle, angular_velocity)
# plt.xlabel("angle")
# plt.ylabel("angular velocity")
# plt.show()


