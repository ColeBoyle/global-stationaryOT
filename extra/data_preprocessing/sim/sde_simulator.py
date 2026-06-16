import numpy as np
from tqdm import tqdm 
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import scanpy as sc
from functools import partial

class Simulation:

    def __init__(self, potential=None, drift=None, 
                 birth=None, death=None, diff_coef=None,
                 sde_sim_parameters={}, save_sim=False) -> None:

        self.potential = potential
        if potential is not None and drift is None:
            self.drift = jax.jit(jax.vmap(jax.grad(potential), in_axes=(0, None))) 
        else:
            self.drift = jax.jit(jax.vmap(drift, in_axes=(0, None)))

        if birth is None:
            birth = lambda x, t: 0 # No birth
        if death is None:
            death = lambda x, t: 0 # No death
        if diff_coef is None and sde_sim_parameters.get("sigma^2", 0) == 0:
            diff_coef = lambda x, t: 0 # No diffusion
        elif diff_coef is None:
            diff_coef = lambda x, t: jnp.sqrt(sde_sim_parameters["sigma^2"]) * jnp.ones_like(x) # Constant diffusion

        self.diff_coef_vmap = jax.jit(jax.vmap(diff_coef, in_axes=(0, None)))

        self.birth_spacevmap = jax.jit(jax.vmap(birth, in_axes=(0, None)))
        self.death_spacevmap = jax.jit(jax.vmap(death, in_axes=(0, None)))

        self.birth_spacetimevmap = jax.jit(jax.vmap(birth, in_axes=(0, 0)))
        self.death_spacetimevmap = jax.jit(jax.vmap(death, in_axes=(0, 0)))

        self.save_sim = save_sim

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


    def sim_sde_jit(self, key, X0, t0, fixed_birth=False, fixed_point=None, sim_birth=True, max_pop=5_000, max_num_itr=1_000,
                    ensure_pos=False):

        if sim_birth:
            birth = self.birth_spacevmap
        else:
            birth = jax.vmap(lambda x, t: 0.0, in_axes=(0, None))

        death = self.death_spacevmap
        
        time_step = self.sde_sim_parameters["time_step"]
        dim = int(self.sde_sim_parameters["dim"])
        #sigma = jnp.sqrt(self.sde_sim_parameters["sigma^2"])
        sigma = self.diff_coef_vmap
        
        N0 = X0.shape[0]
        
        X_padded = jnp.pad(X0, ((0, max_pop - N0), (0, 0)), constant_values=jnp.nan)
        is_dead_col = jnp.concatenate([jnp.zeros((N0, 1)), jnp.ones((max_pop - N0, 1))], axis=0)
        lineage_col = jnp.concatenate([jnp.arange(N0, dtype=jnp.float32)[:, None], 
                                       jnp.full((max_pop - N0, 1), -1.0)], axis=0)
        
        # cell id column (starts identical to lineage for X0)
        individual_col = jnp.concatenate([jnp.arange(N0, dtype=jnp.float32)[:, None], 
                                          jnp.full((max_pop - N0, 1), -1.0)], axis=0)
        
        # X_aug shape is now (max_pop, dim + 3)
        X_aug = jnp.concatenate([X_padded, is_dead_col, lineage_col, individual_col], axis=1)

        def scan_step(carry, i):
            X_now, key, next_id = carry
            t = t0 + i * time_step

            key, subkey_sde, subkey_d, subkey_b = random.split(key, 4)

            X_spat = X_now[:, :dim]
            is_dead = X_now[:, dim]
            lineage = X_now[:, dim+1]
            individual = X_now[:, dim+2]
            is_alive = (is_dead == 0.0)

            drift_val = jnp.where(is_alive[:, None], self.drift(X_spat, t), 0.0)
            noise = random.normal(subkey_sde, shape=(max_pop, dim))
            spatial_step = time_step * drift_val + sigma(X_spat, t) * jnp.sqrt(time_step) * noise

            X_next_spat = jnp.where(is_alive[:, None], X_spat + spatial_step, X_spat)

            if ensure_pos:
                X_next_spat = jnp.where(X_next_spat < 0, X_spat, X_next_spat) # BoolODE handling of negative values

            if fixed_birth:
                X_next_spat = X_next_spat.at[:N0].set(X0)
            if fixed_point is not None:
                fp = fixed_point(t)
                X_next_spat = X_next_spat.at[:len(fp)].set(fp)

            g = jnp.where(is_alive, birth(X_next_spat, t) - death(X_next_spat, t), 0.0)

            # DEATHS 
            roll_d = random.uniform(subkey_d, shape=(max_pop,))
            death_cond = is_alive & (g < 0) & (roll_d < jnp.abs(g) * time_step)
            is_dead_next = jnp.where(death_cond, 1.0, is_dead)

            # BIRTHS
            roll_b = random.uniform(subkey_b, shape=(max_pop,))
            birth_cond = is_alive & (g > 0) & (roll_b < g * time_step)
            num_births = jnp.sum(birth_cond)
            
            parent_indices = jnp.where(birth_cond, jnp.arange(max_pop), max_pop)
            sorted_parents = jnp.sort(parent_indices) 
            
            available_slots = jnp.where(is_dead == 1.0, jnp.arange(max_pop), max_pop)
            sorted_slots = jnp.sort(available_slots) 
            
            birth_mask = jnp.arange(max_pop) < num_births
            
            X_next_combined = jnp.concatenate([X_next_spat, lineage[:, None], individual[:, None]], axis=1)

            new_particles = jnp.where(
                birth_mask[:, None], 
                X_next_combined[sorted_parents % max_pop], 
                jnp.nan
            )

            new_ids = next_id + jnp.arange(max_pop)
            new_particles = new_particles.at[:, dim+1].set(
                jnp.where(birth_mask, new_ids, jnp.nan)
            )
            
            target_indices = jnp.where(birth_mask, sorted_slots % max_pop, -1) 
            X_next_combined = jnp.where(
                (jnp.arange(max_pop) == target_indices[:, None]).any(axis=0)[:, None] & birth_mask.any(),
                new_particles[jnp.argmax(jnp.arange(max_pop) == target_indices[:, None], axis=0)],
                X_next_combined
            )
            
            X_next_spat = X_next_combined[:, :dim]
            lineage_next = X_next_combined[:, dim]
            individual_next = X_next_combined[:, dim+1]
            
            is_dead_next = jnp.where(
                (jnp.arange(max_pop) == target_indices[:, None]).any(axis=0) & birth_mask.any(),
                0.0,
                is_dead_next
            )

            X_next_aug = jnp.concatenate([
                X_next_spat, 
                is_dead_next[:, None], 
                lineage_next[:, None], 
                individual_next[:, None]
            ], axis=1)

            current_pop = jnp.sum(is_dead_next == 0.0)
            freeze_cond = (current_pop == 0) | (current_pop >= max_pop)
            X_next_aug = jnp.where(freeze_cond, X_now, X_next_aug)

            return (X_next_aug, key, next_id + num_births), X_now

        iterations = jnp.arange(0, max_num_itr)
        scan_fn = jax.jit(lambda c, xs: jax.lax.scan(scan_step, c, xs))

        (_, _, _), trajectory_aug = scan_fn((X_aug, key, N0), iterations)

        sampled_trajectory = trajectory_aug 
       
        return sampled_trajectory
        


    def sim_sde_sampled_jit(self, X0, key, t0, verbose=False, fixed_birth=False, fixed_point=None, ensure_pos=False):
        birth = self.birth_spacevmap
        death = self.death_spacevmap
         
        time_step = self.sde_sim_parameters["time_step"]
        dim = int(self.sde_sim_parameters["dim"])
#        sigma = jnp.sqrt(self.sde_sim_parameters["sigma^2"])
        sigma = self.diff_coef_vmap
        max_num_itr = int(self.sde_sim_parameters["max_num_iter"])
        max_pop = int(self.sde_sim_parameters["max_pop"])
        sample_rate = int(self.sde_sim_parameters["sample_rate"])
        
        num_samples = max_num_itr // sample_rate
        
        self.X0 = X0 

        N0 = X0.shape[0]
        

        is_initially_dead = jnp.isnan(X0[:, 0])
        X_padded = jnp.pad(X0, ((0, max_pop - N0), (0, 0)), constant_values=jnp.nan)
        
        is_dead_X0 = jnp.where(is_initially_dead, 1.0, 0.0)[:, None]
        is_dead_col = jnp.concatenate([is_dead_X0, jnp.ones((max_pop - N0, 1))], axis=0)
        
        init_ids = jnp.where(is_initially_dead, -1.0, jnp.arange(N0, dtype=jnp.float32))[:, None]
        lineage_col = jnp.concatenate([init_ids, jnp.full((max_pop - N0, 1), -1.0)], axis=0)
        individual_col = jnp.concatenate([init_ids, jnp.full((max_pop - N0, 1), -1.0)], axis=0)
        
        X_aug = jnp.concatenate([X_padded, is_dead_col, lineage_col, individual_col], axis=1)
        def micro_step(carry, _):
            X_now, key, next_id, current_t = carry

            key, subkey_sde, subkey_d, subkey_b = random.split(key, 4)

            X_spat = X_now[:, :dim]
            is_dead = X_now[:, dim]
            lineage = X_now[:, dim+1]
            individual = X_now[:, dim+2]
            is_alive = (is_dead == 0.0)

            drift_val = jnp.where(is_alive[:, None], self.drift(X_spat, current_t), 0.0)
            noise = random.normal(subkey_sde, shape=(max_pop, dim))
            spatial_step = time_step * drift_val + sigma(X_spat, current_t) * jnp.sqrt(time_step) * noise
            X_next_spat = jnp.where(is_alive[:, None], X_spat + spatial_step, X_spat)
    
            if ensure_pos:
                X_next_spat = jnp.where(X_next_spat < 0, X_spat, X_next_spat) # BoolODE handling of negative values

            if fixed_birth:
                X_next_spat = X_next_spat.at[:N0].set(X0)
            if fixed_point is not None:
                fp = fixed_point(current_t)
                X_next_spat = X_next_spat.at[:len(fp)].set(fp)

            g = jnp.where(is_alive, birth(X_next_spat, current_t) - death(X_next_spat, current_t), 0.0)

            # DEATHS
            roll_d = random.uniform(subkey_d, shape=(max_pop,))
            death_cond = is_alive & (g < 0) & (roll_d < jnp.abs(g) * time_step)
            is_dead_next = jnp.where(death_cond, 1.0, is_dead)

            # BIRTHS
            roll_b = random.uniform(subkey_b, shape=(max_pop,))
            birth_cond = is_alive & (g > 0) & (roll_b < g * time_step)
            num_births = jnp.sum(birth_cond)
            
            parent_indices = jnp.where(birth_cond, jnp.arange(max_pop), max_pop)
            sorted_parents = jnp.sort(parent_indices) 
            
            available_slots = jnp.where(is_dead == 1.0, jnp.arange(max_pop), max_pop)
            sorted_slots = jnp.sort(available_slots) 
            
            birth_mask = jnp.arange(max_pop) < num_births
            
            X_next_combined = jnp.concatenate([X_next_spat, lineage[:, None], individual[:, None]], axis=1)

            new_particles = jnp.where(
                birth_mask[:, None], 
                X_next_combined[sorted_parents % max_pop], 
                jnp.nan
            )
            
            new_ids = next_id + jnp.arange(max_pop)
            new_particles = new_particles.at[:, dim+1].set(
                jnp.where(birth_mask, new_ids, jnp.nan)
            )
            
            target_indices = jnp.where(birth_mask, sorted_slots % max_pop, -1)
            
            X_next_combined = jnp.where(
                (jnp.arange(max_pop) == target_indices[:, None]).any(axis=0)[:, None] & birth_mask.any(),
                new_particles[jnp.argmax(jnp.arange(max_pop) == target_indices[:, None], axis=0)],
                X_next_combined
            )
            
            X_next_spat = X_next_combined[:, :dim]
            lineage_next = X_next_combined[:, dim]
            individual_next = X_next_combined[:, dim+1]
            
            is_dead_next = jnp.where(
                (jnp.arange(max_pop) == target_indices[:, None]).any(axis=0) & birth_mask.any(),
                0.0,
                is_dead_next
            )

            X_next_aug = jnp.concatenate([
                X_next_spat, 
                is_dead_next[:, None], 
                lineage_next[:, None], 
                individual_next[:, None]
            ], axis=1)

            current_pop = jnp.sum(is_dead_next == 0.0)
            freeze_cond = (current_pop == 0) | (current_pop >= max_pop)
            X_next_aug = jnp.where(freeze_cond, X_now, X_next_aug)

            return (X_next_aug, key, next_id + num_births, current_t + time_step), None


        def macro_step(carry, i):
            carry, _ = jax.lax.scan(micro_step, carry, None, length=sample_rate)
            
            X_now, _, _, _ = carry
            return carry, X_now


        macro_scan_fn = jax.jit(lambda c, xs: jax.lax.scan(macro_step, c, xs))
        
        init_carry = (X_aug, key, N0, t0)
        
        _, sampled_trajectory = macro_scan_fn(init_carry, jnp.arange(num_samples))

        sampled_trajectory = jnp.vstack([X_aug[None, ...], sampled_trajectory])
        sim_time = jnp.arange(t0, t0 + (sampled_trajectory.shape[0]) * (time_step * sample_rate) - 1e-8, time_step * sample_rate)

        
        return sampled_trajectory, sim_time

    def make_sim_adata(self, seed, X0, fixed_birth=True, t0=0, 
                       sample_traj=False, get_fp=False, write_path=None
                       ,ensure_pos=False, **kwargs):

        key = random.PRNGKey(seed)

        key, subkey = random.split(key)

        sampled_data, sim_time = self.sim_sde_sampled_jit(X0, subkey, t0, fixed_birth=fixed_birth, ensure_pos=ensure_pos)

        dim = self.sde_sim_parameters["dim"]
        spatial_data = sampled_data[:, :, :dim]
        space_time_data = jnp.concatenate([spatial_data, jnp.broadcast_to(sim_time[:, None, None], (sim_time.shape[0], spatial_data.shape[1], 1))], axis=2)

        alive_mask = (sampled_data[:, :, dim] == 0.0)
        space_time_data = space_time_data[alive_mask]
        times = np.asarray(space_time_data[:, -1], dtype=np.float32)
        space = np.asarray(space_time_data[:, :-1], dtype=np.float32) 

        assert space.shape[0] == times.shape[0], "Mismatch in number of samples between space and time data after filtering for alive cells."
        
        growth_rate = self.birth_spacetimevmap(space, times) - self.death_spacetimevmap(space, times) 

        adata = sc.AnnData(X=space, obs={"time": times,
                                    'growth_rate': growth_rate})
                                                                       
        adata.obs_names = [ f"cell_{i}" for i in range(adata.n_obs)]
        adata.var_names = [ f"gene_{i}" for i in range(adata.n_vars)]

        adata.uns['true_dt'] = self.sde_sim_parameters['time_step']

#        if get_fp:
#
#            key, subkey = random.split(key)
#            max_num_itr = kwargs.get("max_num_traj_itr", 1_000)
#            max_pop_buffer = kwargs.get("max_pop_buffer", 2_000)
#            self.compute_fate_probabilities(subkey, adata, max_num_itr=max_num_itr, max_pop_buffer=max_pop_buffer)

        if sample_traj:

            key, subkey = random.split(key)
            max_num_itr = kwargs.get("max_num_traj_itr", 1_000)
            max_pop_buffer = kwargs.get("max_pop_buffer", 2_000)
            num_traj_samples = kwargs.get("num_traj_samples", 500)
            self.get_trajectories(subkey, adata, num_traj=num_traj_samples, max_num_itr=max_num_itr, max_pop_buffer=max_pop_buffer, ensure_pos=ensure_pos) 

        if write_path is not None:
            adata.write_h5ad(write_path)

        return adata
    
    @partial(jax.jit, static_argnames=['self', 'n_runs', 'batch_size', 'max_pop', 'type_cond_fn', 'dim', 'max_num_itr', 'num_fates'])
    def _process_batch_jit(self, key, batch_cells, time, n_runs, batch_size, max_pop, type_cond_fn, dim, max_num_itr, num_fates):
        """
        100% GPU-bound batch simulation and fate aggregation.
        Never pulls trajectory data back to the CPU.
        """
        X0 = jnp.repeat(batch_cells, n_runs, axis=0)

        batch_traj = self.sim_sde_jit(key, X0, t0=time, fixed_birth=False, fixed_point=None, max_pop=max_pop, max_num_itr=max_num_itr)
        
        T = batch_traj.shape[0]
        
        time_idx = jnp.arange(T)[:, None, None] 
        time_idx = jnp.broadcast_to(time_idx, (T, max_pop, 1))
        traj_with_time = jnp.concatenate([batch_traj, time_idx], axis=-1)
        
        flat_traj = traj_with_time.reshape(T * max_pop, dim + 4)
        
        t_coords = flat_traj[:, :dim]
        t_is_dead = flat_traj[:, dim]
        t_lineage = flat_traj[:, dim+1]
        t_individual = flat_traj[:, dim+2]
        t_time_idx = flat_traj[:, dim+3]
        
        valid_mask = (t_is_dead == 0.0) & (t_individual != -1.0)
        sort_individual = jnp.where(valid_mask, t_individual, -1.0)
        
        sort_idx = jnp.lexsort((t_time_idx, sort_individual))
        
        sorted_coords = t_coords[sort_idx]
        sorted_individual = sort_individual[sort_idx]
        sorted_lineage = t_lineage[sort_idx]
        sorted_valid = valid_mask[sort_idx]
        
        id_changes = jnp.concatenate([
            sorted_individual[:-1] != sorted_individual[1:], 
            jnp.array([True])
        ])
        
        is_terminal = sorted_valid & id_changes
        
        parent_indices = jnp.where(sorted_lineage >= 0, (sorted_lineage // n_runs).astype(jnp.int32), -1)
        
        types_all = jnp.where(is_terminal, type_cond_fn(sorted_coords[:, :dim], time), -1)
        
        one_hot_fates = jax.nn.one_hot(types_all, num_classes=num_fates)
        
        one_hot_fates = jnp.where(is_terminal[:, None], one_hot_fates, 0.0)
        
        fates = jax.ops.segment_sum(one_hot_fates, parent_indices, num_segments=batch_size)
        
        total_surviving = jax.ops.segment_sum(is_terminal.astype(jnp.int32), parent_indices, num_segments=batch_size)
        
        return fates, total_surviving


    def compute_fate_probabilities(self, key, adata, type_cond_fn=None, type_label='cell_type',  time_col='time', max_num_itr=1000, max_pop_buffer=2000):

            sim_time = np.unique(adata.obs[time_col])
            dim = adata.X.shape[1]
            all_labels = np.unique(adata.obs[type_label]) 
            num_fates = len(all_labels)
            for label in all_labels:
                adata.obs[label + '_fp'] = np.nan

            adata.obsm[type_label + '_fp'] = np.full((adata.n_obs, len(all_labels)), np.nan)

            n_runs = 500
            batch_size = 50

            print(f"Computing fate probabilities with {n_runs} simulations per cell, in batches of {batch_size}...")
            
            static_max_pop = (batch_size * n_runs) + max_pop_buffer
    
            for time in tqdm(sim_time):
                cells_t = adata[adata.obs[time_col] == time].X
                cell_names = adata[adata.obs[time_col] == time].obs_names
                
                for i in range(0, cells_t.shape[0], batch_size):
                    batch_cells = cells_t[i:i+batch_size]
                    cell_names_batch = cell_names[i:i+batch_size]
                    cell_int_indices = adata.obs.index.get_indexer(cell_names_batch)
                    current_batch_len = batch_cells.shape[0]
                    
                    if current_batch_len < batch_size:
                        pad_size = batch_size - current_batch_len
                        batch_cells_padded = jnp.vstack([batch_cells, jnp.zeros((pad_size, dim))])
                    else:
                        batch_cells_padded = batch_cells

                    key, subkey = random.split(key)
                    
                    fps, total_surv = self._process_batch_jit(
                        subkey,
                        jnp.array(batch_cells_padded), 
                        time=time, 
                        n_runs=n_runs, 
                        batch_size=batch_size, 
                        max_pop=static_max_pop, 
                        type_cond_fn=type_cond_fn, 
                        dim=dim,
                        max_num_itr=max_num_itr,
                        num_fates=num_fates
                    )
                    
                    fps = np.array(fps[:current_batch_len])
                    total_surv = np.array(total_surv[:current_batch_len])
                    

                    safe_denom = jnp.where(total_surv == 0, 1, total_surv)

                    fps = fps / safe_denom[:, None]  

                    for idx, label in enumerate(all_labels):
                        adata.obs.loc[cell_names_batch, label + '_fp'] = fps[:, idx]

                    adata.obsm[type_label + '_fp'][cell_int_indices] = fps
                    

    def get_trajectories(self, key, adata, num_traj=500, time_col='time', growth_rate_col='growth_rate', max_num_itr=1000, max_pop_buffer=2000, ensure_pos=False):

        unique_times = np.unique(adata.obs[time_col])
        trajectories = {}
        trajectories_metadata = {}
        print(f"Extracting {num_traj} trajectories starting at each of {len(unique_times)} time points...")

        for time in tqdm(unique_times):
            growth_ = adata.obs.loc[adata.obs[time_col] == time, growth_rate_col].values
            growth_ = growth_[growth_ > 0]
            growth_dist = growth_ / growth_.sum()
            key, subkey = random.split(key)

            data_str_ind = adata.obs.loc[(adata.obs[time_col] == time) & (adata.obs[growth_rate_col] > 0)].index
            data_int_ind = adata.obs.index.get_indexer(data_str_ind)

            X0_ind = random.choice(subkey, a=data_int_ind, 
                                shape=(num_traj,), p=growth_dist, replace=True)
            
            X0 = adata.X[X0_ind]
            key, subkey = random.split(key)
            traj = self.sim_sde_jit(subkey, X0, t0=time, fixed_birth=False, sim_birth=False, 
                                    max_num_itr=max_num_itr, max_pop=num_traj + max_pop_buffer,
                                    ensure_pos=ensure_pos)
            traj, _, traj_len, traj_lin = extract_trajectories_fast(traj)
            
            
            max_traj_len = traj.shape[1]
            # get number of trajectories that hit the max_num_itr limit (i.e., potentially incomplete trajectories)
            num_incomplete = np.sum(traj_len >= max_num_itr)

            # warn if more than 5% of trajectories hit the max_num_itr limit
            if num_incomplete / num_traj > 0.05:
                print(f"Warning: {num_incomplete} out of {num_traj} trajectories (>{(num_incomplete / num_traj) * 100:.1f}%) hit the max_num_itr limit of {max_num_itr}. Consider increasing max_num_itr for more complete trajectories.")

            trajectories[str(time)] = np.asarray(traj, np.float32)
            trajectories_metadata[str(time)] = {
                'lengths': np.asarray(traj_len, np.int32),
                'lineages': np.asarray(traj_lin, np.int32)
            }

        adata.uns['true_traj_data'] = trajectories
        adata.uns['true_traj_metadata'] = trajectories_metadata

# extract individual trajectories from the augmented trajectory data 
def extract_trajectories_fast(trajectory_aug):

    traj_data = np.asarray(trajectory_aug)
    
    if traj_data.shape[0] > 1:
        is_same = (traj_data[1:] == traj_data[:-1]) | (np.isnan(traj_data[1:]) & np.isnan(traj_data[:-1]))
        
        step_identical = np.all(is_same, axis=(1, 2))
        freeze_indices = np.where(step_identical)[0]
        
        if len(freeze_indices) > 0:
            first_freeze = freeze_indices[0]
            traj_data = traj_data[:first_freeze + 1]
    
    dim = traj_data.shape[-1] - 3
    is_dead_idx = dim
    lineage_idx = dim + 1
    individual_idx = dim + 2
    
    active_mask = (traj_data[:, :, is_dead_idx] == 0.0) & (traj_data[:, :, individual_idx] != -1.0)
    times, slots = np.where(active_mask)
    
    if len(times) == 0:
        return np.zeros((0, 0, dim)), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        
    valid_ids = traj_data[times, slots, individual_idx]
    valid_coords = traj_data[times, slots, :dim]
    valid_lineages = traj_data[times, slots, lineage_idx] 
    
    sort_idx = np.lexsort((times, valid_ids))
    
    sorted_ids = valid_ids[sort_idx]
    sorted_times = times[sort_idx]
    sorted_coords = valid_coords[sort_idx]
    sorted_lineages = valid_lineages[sort_idx] 
    
    id_changes = np.where(sorted_ids[:-1] != sorted_ids[1:])[0] + 1
    
    traj_list = np.split(sorted_coords, id_changes)
    time_list = np.split(sorted_times, id_changes)
    lineage_list = np.split(sorted_lineages, id_changes) 
    
    n_traj = len(traj_list)
    max_traj_len = max(len(t) for t in traj_list)
    
    padded_matrix = np.zeros((n_traj, max_traj_len, dim))
    creation_times = np.zeros(n_traj, dtype=int)
    traj_lineages = np.zeros(n_traj, dtype=int) 
    
    traj_lens = []
    for i, (traj, t_arr, lin_arr) in enumerate(zip(traj_list, time_list, lineage_list)):
        L = len(traj)
        traj_lens.append(L)
        padded_matrix[i, :L, :] = traj
        
        if L < max_traj_len:
            padded_matrix[i, L:, :] = traj[-1, :]
            
        creation_times[i] = t_arr[0]
        traj_lineages[i] = lin_arr[0] 
        
    traj_lens = np.array(traj_lens)
    
    sort_idx = np.argsort(creation_times)
    padded_matrix = padded_matrix[sort_idx]
    creation_times = creation_times[sort_idx]
    traj_lens = traj_lens[sort_idx]
    traj_lineages = traj_lineages[sort_idx] 
    
    return padded_matrix, creation_times, traj_lens, traj_lineages