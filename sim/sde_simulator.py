import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm 
import numpy as np
import jax
import jax.numpy as jnp
from jax import random


class Simulation:

    def __init__(self, potential=None, drift=None, birth=None, death=None, sde_sim_parameters={}, save_sim=False) -> None:

        self.potential_ = potential
        if potential is not None and drift is None:
            self.drift = jax.jit(jax.vmap(jax.grad(potential), in_axes=(0, None))) 
        else:
            self.drift = drift

        if birth is None:
            birth = lambda x, t: 0 # No birth
        if death is None:
            death = lambda x, t: 0 # No death


        self.birth = birth
        self.death = death 

        self.growth = lambda x, t: birth(x, t) - death(x, t)

        self.save_sim = save_sim

        # SDE Simulation parameters
        if not sde_sim_parameters:

            sde_sim_parameters = {
                "sigma^2": 0.15,
                "time_step": 0.001,
                "dim": 2,
                "max_num_iter": 10e3,
                "max_pop": 2e4,
                "sample_rate": 100,
                "print_rate": 100,
                "seed": 123
            }

        self.sde_sim_parameters = sde_sim_parameters

    def write_sim_time_series(self, file_name="sim_time_series"):
        np.save(self.save_folder + f"/{file_name}.npy", self.sim_time_series, allow_pickle=True)

    
    def set_sde_sim_parameters(self, sde_sim_parameters):
        self.sde_sim_parameters = sde_sim_parameters

    def sim_sde(self, X0, t0, verbose=False, fixed_birth=False, exact=False, fixed_point=None):
   
        birth = self.birth
        death = self.death
        seed = self.sde_sim_parameters["seed"]
        
        time_step = self.sde_sim_parameters["time_step"]
        dim = self.sde_sim_parameters["dim"]
        sigma = np.sqrt(self.sde_sim_parameters["sigma^2"])
        max_num_itr = int(self.sde_sim_parameters["max_num_iter"])
        max_pop = self.sde_sim_parameters["max_pop"]
        print_rate = self.sde_sim_parameters["print_rate"]
        sample_rate = self.sde_sim_parameters["sample_rate"] 
        
        self.X0 = X0 
        key = random.PRNGKey(seed)
        dist_ts = [X0]
        pop_size = [jnp.count_nonzero(jnp.isnan(X0))]
        source_pop_size = [jnp.count_nonzero(jnp.where((birth(X0, 0) - death(X0, 0)) > 0, 1, 0))]
        if exact:
            data = jnp.full((int(max_num_itr), int(max_pop), dim), jnp.nan)
            data = data.at[0,:len(X0),:].set(X0)
        else:
            data = X0
    
        def update(i, data, key, exact=False):

            t = t0 + i * time_step

            if exact:
                X_now = data[i-1]
            else:
                X_now = data

            N = len(X_now)

            key, subkey = random.split(key)

            X_next = X_now + time_step * self.drift(X_now, t) + sigma * np.sqrt(time_step) * random.normal(subkey, shape=(N, dim))

            if fixed_birth:
                X_next = X_next.at[:len(X0)].set(X0)

            if fixed_point is not None:
                fp = fixed_point(t)
                X_next = X_next.at[:len(fp)].set(fp)

            growth = birth(X_next, t) - death(X_next, t) 

            if exact:
                # death
                key, subkey = random.split(key)
                roll = random.uniform(subkey, shape=(len(growth),))
                T = jnp.where((growth < 0) & (roll < jnp.abs(growth) * time_step), True, False)
                X_next = X_next.at[T].set(jnp.nan)
                
                # growth 
                key, subkey = random.split(key)
                roll = random.uniform(subkey, shape=(len(growth),))
                G = jnp.where((growth > 0) & (roll < growth * time_step), True, False)
    
                lst_idx = jnp.argmax(jnp.where(jnp.isnan(jnp.flip(X_next[:,0])), 0, 1))
                lst_idx = len(X_next) - lst_idx
    
                n = jnp.count_nonzero(G)
                X_next = X_next.at[lst_idx:lst_idx + n].set(X_next[G])
                data = data.at[i,:, :].set(X_next)                              

            else:
                key, subkey = random.split(key)
                roll = random.uniform(subkey, shape=(len(growth),))
                T = jnp.where((growth < 0) & (roll < jnp.abs(growth) * time_step), True, False)

                # Remove dead particles X_next is numpy array
                X_next = jnp.delete(X_next, T, axis=0)
                growth = jnp.delete(growth, T)
                

                key, subkey = random.split(key)
                roll = random.uniform(subkey, shape=(len(growth),))
                G = jnp.where((growth > 0) & (roll < growth * time_step), True, False)

                X_next = jnp.append(X_next, X_next[G], axis=0)

                return X_next


            return data

        for i in tqdm(range(1, int(max_num_itr-1)), disable=not verbose):

            if exact:
                key, subkey = random.split(key) 
                if i > data.shape[0]:
                    if verbose:
                        print("Reached max number of iterations", i)
                    max_num_itr = i-1
                    break

                if jnp.all(jnp.isnan(data[i-1].flatten())):
                    if verbose:
                        print("Population reached 0")
                    max_num_itr = i-1
                    data = data[:i-1]
                    break

                data = update(i, data, subkey, exact=exact)

            else:
                key, subkey = random.split(key)    

                if len(data) >= max_pop:
                    max_num_itr = i
                    if verbose:
                        print("Population reached the max_pop of ", max_pop)
                    break

                if len(data) == 0:
                    if verbose:
                        print("Population reached 0")
                    max_num_itr = i
                    break

                data = update(i, data, subkey, exact=exact)

                if i>0 and i % sample_rate == 0:
                    dist_ts += [data]
                    jax.clear_caches()
                    if verbose & (i % print_rate == 0):
                        print("Itration:", i, "N =", len(data))


        if not exact:
            data = dist_ts 

        self.source_pop_size_series = source_pop_size
        self.pop_size_series = pop_size
        self.sim_time_series = data 

        self.sim_time = jnp.linspace(t0, t0 + (data.shape[0]) * time_step, data.shape[0]) if exact else jnp.arange(t0, t0 + max_num_itr * time_step, time_step * sample_rate)

