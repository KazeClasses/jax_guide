import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Int, PRNGKeyArray
from generate_data import PendulumSimulation
import optax

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
    
model = CNNEmulator(jax.random.PRNGKey(0))
pendulum = PendulumSimulation(image_size=64)
dataset = pendulum.generate_dataset(5, 9.8, 1.0)

def loss_fn(model, batch):
    x, y = batch
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)

def train(
    model: CNNEmulator,
    dataset: Float[Array, " n_samples n_res n_res"],
    batch_size: Int,
    learning_rate: Float,
    num_epochs: Int,
    key: PRNGKeyArray,
) -> CNNEmulator:
    
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: CNNEmulator,
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

trained_model = train(model, dataset, 4, 1e-3, 300, jax.random.PRNGKey(0))
result = trained_model(dataset[0][0])