import jax
import jax.numpy as jnp
from functools import partial
import yourdfpy
import pyroki as pk

def load_robot(urdf_path):
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)  
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    return robot, robot_coll

@partial(jax.jit, static_argnames=["func"])
def compute_points(func, num):
    ts = jnp.linspace(0, 1, num)
    bat_func = jax.vmap(func, in_axes=(0))
    return bat_func(ts)
    
@jax.jit
def jacobian_proj(cf, steps, q, robot):
    J = jax.jacobian(cf, 0)
    def body(i, val):
        qk = val
        J_dag = jnp.linalg.pinv(jnp.array([J(qk, robot)]))
        dx = jnp.array([-cf(qk, robot)])
        dq = J_dag @ dx
        return qk + dq
    initial = q
    q = jax.lax.fori_loop(0, steps, body, initial)
    return q
