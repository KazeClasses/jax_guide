import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dx
from jaxtyping import Float, Array, Int, PRNGKeyArray
from generate_data import PendulumSimulation
import optax

class latent_term(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)

class LatentODE(eqx.Module):

    encoder: list
    decoder: list
    # ode_term: dx.ODETerm
    n_hidden: Int
    n_layers: Int
    n_bottle_pixels: Int
    n_bottle_channels: Int

    def __init__(
        self,
        key: PRNGKeyArray,
        image_size: Int,
        n_hidden: Int = 4,
        n_layers: Int = 2,
    ):
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        encoder = []
        decoder = []

        key, subkey = jax.random.split(key)
        encoder.append(eqx.nn.Conv2d(2, n_hidden, 3, padding=[1,1], key=subkey))
        encoder.append(eqx.nn.Lambda(jnp.tanh))
        for i in range(n_layers):
            key, subkey = jax.random.split(key)
            encoder.append(eqx.nn.Conv2d(n_hidden, n_hidden, 3, padding=[1,1], key=subkey))
            encoder.append(eqx.nn.Lambda(jnp.tanh))
        self.encoder = encoder

        test_image = jnp.zeros((2, image_size, image_size))
        bottle_dim = self.encode(encoder, test_image)
        self.n_bottle_channels = bottle_dim.shape[0]
        self.n_bottle_pixels = bottle_dim.shape[1]

        key, subkey = jax.random.split(key)
        for i in range(n_layers):
            key, subkey = jax.random.split(key)
            decoder.append(eqx.nn.Conv2d(n_hidden, n_hidden, 3, padding=[1,1], key=subkey))
            decoder.append(eqx.nn.Lambda(jnp.tanh))

        key, subkey = jax.random.split(key)
        decoder.append(eqx.nn.ConvTranspose2d(n_hidden, 1, 3, padding=[1,1], key=subkey))
        self.decoder = decoder

    def __call__(
        self, x: Float[Array, "2 n_res n_res"]
    ) -> Float[Array, "1 n_res n_res"]:
        x = self.encode(self.encoder, x)
        return self.decode(x)
    
    @staticmethod
    def encode(model: list, x: Float[Array, "2 n_res n_res"]) -> Float[Array, " n_bottleneck"]:
        for layer in model:
            x = layer(x)
        return x

    def decode(self, z: Float[Array, " n_bottleneck"]) -> Float[Array, " n_res n_res"]:
        z = z.reshape((self.n_hidden, self.n_bottle_pixels, self.n_bottle_pixels))
        for i in range(self.n_layers):
            z = self.decoder[2*i](z)
            z = self.decoder[1 + 2*i](z)
        z = self.decoder[-2](z)
        z = self.decoder[-1](z)
        return z


model = LatentODE(jax.random.PRNGKey(0), 64)
pendulum = PendulumSimulation(image_size=64)
dataset = pendulum.generate_dataset(5, 9.8, 1.0)


def loss_fn(model, batch):
    x, y = batch
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)

def train(
    model: LatentODE,
    dataset: Float[Array, " n_samples n_res n_res"],
    batch_size: Int,
    learning_rate: Float,
    num_epochs: Int,
    key: PRNGKeyArray,
) -> LatentODE:
    
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: LatentODE,
        opt_state: optax.OptState,
        batch: Float[Array, " n_samples n_res n_res"],
    ) -> tuple:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    print("Training...")
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, jnp.arange(len(dataset)))
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[0][indices[i:i+batch_size]], dataset[1][indices[i:i+batch_size]]]
            model, opt_state, loss = make_step(model, opt_state, batch)
            print(f"Epoch {epoch}, Loss: {loss}")
    return model

trained_model = train(model, dataset, 4, 5e-3, 1000, jax.random.PRNGKey(0))
result = trained_model(dataset[0][0])