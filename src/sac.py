import os, sys, time, json, pickle, numpy as np
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jax, jax.numpy as jnp, optax
from sim import *

OBS = 22

def init_mlp(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        lim = np.sqrt(6 / (a + b)) * (0.01 if i == len(sizes) - 2 else 1.0)
        ps.append((jax.random.uniform(k, (a, b), minval=-lim, maxval=lim), jnp.zeros(b)))
    return ps

def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = ps[-1]
    return x @ W + b

NEG = -1e9

def masked_logp(logits, m):
    lg = jnp.where(m, logits, NEG)
    return jax.nn.log_softmax(lg), jax.nn.softmax(lg)

def make_update(gamma, tau, target_H, lr):
    opt = optax.adam(lr)
    def update(state, batch):
        (actor, q1, q2, q1t, q2t, log_a), (oa, o1, o2, oal) = state
        o, a, r, o2_, d, m, m2 = batch
        alpha = jnp.exp(log_a)
        lp2, p2 = masked_logp(mlp(actor, o2_), m2)
        qt = jnp.minimum(mlp(q1t, o2_), mlp(q2t, o2_))
        V = jnp.sum(jnp.where(m2, p2 * (qt - alpha * lp2), 0.0), -1)
        y = jax.lax.stop_gradient(r + gamma * (1 - d) * V)
        def qloss(q):
            qa = jnp.take_along_axis(mlp(q, o), a[:, None], 1)[:, 0]
            return jnp.mean((qa - y) ** 2)
        l1, g1 = jax.value_and_grad(qloss)(q1); l2, g2 = jax.value_and_grad(qloss)(q2)
        u1, o1 = opt.update(g1, o1, q1); q1 = optax.apply_updates(q1, u1)
        u2, o2 = opt.update(g2, o2, q2); q2 = optax.apply_updates(q2, u2)
        qmin = jax.lax.stop_gradient(jnp.minimum(mlp(q1, o), mlp(q2, o)))
        def aloss(actor):
            lp, p = masked_logp(mlp(actor, o), m)
            ent = -jnp.sum(jnp.where(m, p * lp, 0.0), -1)
            return jnp.mean(jnp.sum(jnp.where(m, p * (alpha * lp - qmin), 0.0), -1)), ent
        (la, ent), ga = jax.value_and_grad(aloss, has_aux=True)(actor)
        ua, oa = opt.update(ga, oa, actor); actor = optax.apply_updates(actor, ua)
        H_ = jax.lax.stop_gradient(jnp.mean(ent))
        gl = H_ - target_H   # d/dlog_a of log_a*(H - target)
        ul, oal = opt.update(gl, oal, log_a); log_a = optax.apply_updates(log_a, ul)
        q1t = jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, q1t, q1)
        q2t = jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, q2t, q2)
        return ((actor, q1, q2, q1t, q2t, log_a), (oa, o1, o2, oal)), (l1, la, H_, alpha)
    return jax.jit(update), opt

@jax.jit
def act_sample(actor, o, m, key):
    lg = jnp.where(m, mlp(actor, o), NEG)
    return jax.random.categorical(key, lg)

@jax.jit
def act_greedy(actor, o, m):
    return jnp.argmax(jnp.where(m, mlp(actor, o), NEG), -1)

def assign(dst, rows, src, idx):
    for f in Pop.FIELDS:
        getattr(dst, f)[rows] = getattr(src, f)[idx]
    dst.c_hist[rows] = src.c_hist[idx]

def train(P, pool, w, seed, iters=40000, n_env=16, batch=256, gamma=0.99, tau=0.005, target_frac=0.5,
          lr=3e-4, warmup=20000, buf=300000, log=None, r_center=0.55):
    rng = np.random.default_rng([seed, 42])
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 4)
    actor = init_mlp(ks[0], [OBS, 256, 128, 64, N_ACT])
    q1 = init_mlp(ks[1], [OBS, 256, 256, N_ACT]); q2 = init_mlp(ks[2], [OBS, 256, 256, N_ACT])
    target_H = target_frac * np.log(N_ACT)
    update, opt = make_update(gamma, tau, target_H, lr)
    log_a = jnp.array(np.log(0.05))
    state = ((actor, q1, q2, q1, q2, log_a), (opt.init(actor), opt.init(q1), opt.init(q2), opt.init(log_a)))
    B = dict(o=np.zeros((buf, OBS), np.float32), a=np.zeros(buf, np.int32), r=np.zeros(buf, np.float32),
             o2=np.zeros((buf, OBS), np.float32), d=np.zeros(buf, np.float32), m=np.zeros((buf, N_ACT), bool),
             m2=np.zeros((buf, N_ACT), bool))
    ptr = 0; full = False
    env = Pop(n_env); idx = rng.integers(0, pool.n, n_env); assign(env, np.arange(n_env), pool, idx)
    t = np.zeros(n_env, int)
    prev = None
    ret_ep, ep_rets = np.zeros(n_env), []
    t0 = time.time(); key = ks[3]
    for it in range(iters + warmup // n_env):
        r = draws(rng, n_env)
        obs, info = observe(P, env, r, H - t)
        m = masks(env)
        if prev is not None:
            po, pa, pr, pd, pm = prev
            k = n_env; sl = np.arange(ptr, ptr + k) % buf
            B["o"][sl], B["a"][sl], B["r"][sl], B["o2"][sl], B["d"][sl], B["m"][sl], B["m2"][sl] = po, pa, pr, obs, pd, pm, m
            ptr = (ptr + k) % buf; full = full or ptr < k
        if it * n_env < warmup:
            a = np.array([rng.choice(np.where(mm)[0]) for mm in m])
        else:
            key, sk = jax.random.split(key)
            a = np.asarray(act_sample(state[0][0], obs, m, sk))
        dest, ev = step(P, env, a, r)
        rew = (utility(P, env, w, dest >= 0) - r_center).astype(np.float32)
        apply_moves(P, env, dest, r)
        t += 1
        done = (t >= H).astype(np.float32)
        ret_ep += rew
        prev = (obs, a.astype(np.int32), rew, done, m)
        if done.any():
            dn = np.where(done > 0)[0]
            ep_rets += list(ret_ep[dn]); ret_ep[dn] = 0
            assign(env, dn, pool, rng.integers(0, pool.n, len(dn))); t[dn] = 0
        if it * n_env >= warmup:
            nmax = buf if full else ptr
            bi = rng.integers(0, nmax, batch)
            bt = tuple(jnp.asarray(B[k][bi]) for k in ["o", "a", "r", "o2", "d", "m", "m2"])
            state, stats = update(state, bt)
            if log is not None and it % 5000 == 0:
                l1, la, H_, al = [float(x) for x in stats]
                msg = f"seed {seed} it {it} qloss {l1:.3f} H {H_:.3f} alpha {al:.4f} ep_ret {np.mean(ep_rets[-200:]) if ep_rets else 0:.2f} {time.time()-t0:.0f}s"
                print(msg, file=log, flush=True)
    st = jax.tree_util.tree_map(np.asarray, state[0])
    return dict(actor=st[0], q1=st[1], q2=st[2])

@jax.jit
def q_greedy(q1, q2, o, m):
    return jnp.argmax(jnp.where(m, jnp.minimum(mlp(q1, o), mlp(q2, o)), NEG), -1)

def drs_fn_from(net, mode="actor"):
    if mode == "actor":
        return lambda o, m: np.asarray(act_greedy(net["actor"], jnp.asarray(o), jnp.asarray(m)))
    return lambda o, m: np.asarray(q_greedy(net["q1"], net["q2"], jnp.asarray(o), jnp.asarray(m)))
