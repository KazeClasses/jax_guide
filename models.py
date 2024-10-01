import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dx
from jaxtyping import Float, Array, Int, PRNGKeyArray

class CNNEmulator(eqx.Module):
    layers: list

    def __init__(self, key: PRNGKeyArray, hidden_dim: Int = 4):
        layers = []
        layers.append(eqx.nn.Conv2d(2, hidden_dim, 3, padding=[1,1], key = key))
        layers.append(eqx.nn.Lambda(jnp.tanh))
        for i in range(4):
            key, subkey = jax.random.split(key)
            # Take two frames then predict the next one
            layers.append(eqx.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=[1,1], key = subkey))

            layers.append(eqx.nn.Lambda(jnp.tanh))
        key, subkey = jax.random.split(key)
        layers.append(eqx.nn.Conv2d(hidden_dim, 1, 3, padding=[1,1], key = key))
        self.layers = layers

    def __call__(self, x: Float[Array, "2 n_res n_res"]) -> Float[Array, "1 n_res n_res"]:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def rollout(self, x: Float[Array, "2 n_res n_res"], n_step: Int) -> Float[Array, "n_step n_res n_res"]:
        result = [x]
        for i in range(n_step):
            x = jnp.concatenate([x[1:], self(x)], axis=0)
            result.append(x[1:])
        return jnp.stack(result)

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
    
